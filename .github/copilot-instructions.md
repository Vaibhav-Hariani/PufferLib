# Copilot Instructions for PufferLib
## Architecture
- PufferLib wraps heterogeneous environments into a common `PufferEnv` interface (`pufferlib/pufferlib.py`), enforcing `single_observation_space`, `single_action_space`, and shared buffers via `set_buffers`.
- `pufferlib/emulation.py` adapts Gymnasium and PettingZoo envs, flattening observations/actions and rehydrating them with `emulate`/`nativize` to keep zero-copy compatibility.
- Vectorized execution flows through `pufferlib/vector.py`: `Serial` for local loops, `Multiprocessing` for shared-memory workers (zero-copy batches), and `Ray` for distributed runs; all go through the same buffer schema.
- Training orchestration lives in `pufferlib/pufferl.py`, which wires vector envs, Torch policies (`pufferlib/models.py`), and hyperparameter sweeps (`pufferlib/sweep.py`).
- High-performance "Ocean" environments (`pufferlib/ocean/**/`) compile C/Raylib/Box2D backends via `binding.c`; PyTorch kernels and Cython emulation live under `pufferlib/extensions/` and expose `_C` for advantage computation.
## Build & Install
- `pip install -e .` triggers `setup.py` to download Raylib/Box2D and build both the Torch extension (`pufferlib._C`) and any enabled ocean bindings.
- Use environment toggles: `NO_OCEAN=1` to skip C/Raylib downloads (pure Python dev) and `NO_TRAIN=1` to avoid Torch/pyro extras when editing environments only.
- For rebuilds with debug symbols run `DEBUG=1 python setup.py build_ext --inplace --force`; the comment inside `setup.py` shows the matching `LD_PRELOAD` line for AddressSanitizer.
- Standalone ocean binaries are produced with `scripts/build_ocean.sh <env> [local|fast|web]`, which links against the vendored Raylib/Box2D archives.
## Test & QA
- Fast smoke tests cover API contracts: `pytest tests/test_api.py tests/test_nested.py tests/test_flatten.py`.
- Vector backends and env bindings: `pytest tests/test_env_binding.py tests/test_newbind.py` require ocean libs; rerun after touching `pufferlib/vector.py` or any `binding.c`.
- GPU/Torch kernels are exercised by `pytest tests/test_c_advantage.cu tests/test_puffernet.py`; skip or set `CUDA_VISIBLE_DEVICES=` if no GPU is present.
- Registry extras can be validated with `bash tests/test_registry.sh`, which installs each `[extra]` declared under `project.optional-dependencies` before calling `tests/test_registry.py`.
## Code Patterns
- Native `PufferEnv` subclasses should avoid creating new numpy arrays; reuse the allocated buffers from `set_buffers` and write into slices to stay compatible with vector memory sharing.
- Gymnasium wrappers must return infos as a list (single-agent) or dict keyed by agent to satisfy `pufferlib.vector` aggregation; mismatched shapes trip the `check_space` guardrails in `emulation.py`.
- When adding configs, inherit from `config/default.ini` and keep environment-specific overrides in `pufferlib/config/<env>.ini`; the CLI merges these via `ConfigParser`.
- `pufferlib.vector.make` expects callables accepting `buf` and `seed`; propagate unknown kwargs through to keep Serial/Multiprocessing/Ray parity.
## Integration Notes
- Stick to the dtype conversions defined in `pufferlib.pytorch.numpy_to_torch_dtype_dict`; stray `torch.tensor(...)` casts will break the byte-level `nativize_tensor` logic used under torch.compile.
- Multi-agent infos are averaged in `vector.Serial._avg_infos`; include scalar statistics or lists of scalars so aggregations survive batching.
- Ocean bindings expose `binding.env_init/vec_init/vec_step` and expect the same contiguous buffers provided by `PufferEnv`; keep C structs in sync with the Python buffer layout when editing `binding.c`.
- The package exposes a `puffer` console script (`pyproject.toml`); `puffer train <config>` or `python -m pufferlib.pufferl train <config>` is the supported entry point for end-to-end experiments.
