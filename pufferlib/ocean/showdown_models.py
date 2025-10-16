from typing import Tuple

import torch
from torch import nn

# Local replacements for pufferlib utilities so this file is self-contained.
def layer_init(layer: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    """Initialize linear layer weights and biases.

    - weights: orthogonal initialized with given gain (std)
    - bias: constant bias_const
    Returns the layer for convenience.
    """
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, gain=std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        if env:
            self.obs_shape = env.single_observation_space.shape
        else:
            ##Todo: Revert when env is real.
            self.obs_shape = (206,)  # Default to Showdown obs shape (choices removed)
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        #self.pre_layernorm = nn.LayerNorm(hidden_size)
        #self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        hidden = self.policy.encode_observations(observations, state=state)
        h = state['lstm_h']
        c = state['lstm_c']

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
            lstm_state = (h, c)
        else:
            lstm_state = None

        #hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        #hidden = self.post_layernorm(hidden)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x = observations
        lstm_h = state['lstm_h']
        lstm_c = state['lstm_c']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, 'LSTM state must be (h, c)'
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B*TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        #hidden = self.pre_layernorm(hidden)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()
 
        #hidden = self.post_layernorm(hidden)
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B*TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        #state.batch_logits = logits.reshape(B, TT, -1)
        state['hidden'] = hidden
        state['lstm_h'] = lstm_h.detach()
        state['lstm_c'] = lstm_c.detach()
        return logits, values


class ShowdownLSTM(LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256):
        # policy = Showdown(env, hidden_size=hidden_size, depth=depth)
        super().__init__(env, policy, input_size, hidden_size)

class Showdown(nn.Module):
    MAX_MOVE = 165.0
    MAX_POKE = 151.0
    
    def __init__(self, env, input_size=1024, hidden_size=256, depth=2):
        super().__init__()
        self.num_steps = 0
        self.is_continuous = False

        # Packing / shape params (align with showdown_models.Showdown)
        self.embed_size = 5  # Each pokemon row embedding output dim after summation
        self.rows = 13
        self.num_pokemon_rows = self.rows - 1
        self.header_size = 4
        self.pokemon_row_length = 7

        self.stats_per_player = 7
        self.stats_total = self.stats_per_player * 2

        self.active_flag_len = 1
        self.move_pp_len = 4
        self.status_flags = 6

        # Each pokemon is now represented by concatenation of
        # species_emb (5) + 4 move embeddings (4*5=20) + active_flag (1) = 26 dims.
        # We keep move_pp (4) and status flags (6) separate.
        # Per-pokemon unpacked dims = 26 + 4 + 6 = 36
        self.per_pokemon_unpacked = (self.embed_size) * 5 + 1 + self.move_pp_len + self.status_flags

        self.hidden_size = hidden_size
        self.num_actions = 10

        # Compute input_size consistently from components
        self.input_size = self.stats_total + (self.num_pokemon_rows * self.per_pokemon_unpacked)

        # +1 because move/pokemon IDs are 1-based or may include the max id (e.g. STRUGGLE)
        self.species_embed = nn.Embedding(int(self.MAX_POKE) + 1, self.embed_size)
        self.move_embed = nn.Embedding(int(self.MAX_MOVE) + 1, self.embed_size)

        # We no longer project concatenated species+moves down to embed_size.
        # Instead, pokemon_vec will be the raw concatenation of species_emb (5)
        # and the 4 move embeddings (4*5=20) -> 25 dims.

        # First encoder layer: plain projection + GELU (don't use layer_init here
        # so concat_in flows directly through the encoder projection and nonlinearity)
        encoder_layers = [nn.Linear(self.input_size, self.hidden_size), nn.GELU()]
        for i in range(depth):
            encoder_layers.append(layer_init(nn.Linear(self.hidden_size, self.hidden_size), std=1.0))
            encoder_layers.append(nn.GELU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = layer_init(
            nn.Linear(self.hidden_size, self.num_actions), std=0.01
        )
        self.value = layer_init(nn.Linear(self.hidden_size, 1), std=1)


    def forward(self, observations, state=None):
        ##print for the very first batch 
        # ShowdownParser.pretty_print(observations[0].cpu().numpy())
        return self.forward_eval(observations, state)


    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def encode_observations(self, observations: torch.Tensor, state=None):
        unpacked = self.unpack_obs(observations)
        out = self.encoder(unpacked)
        return out

    def decode_actions(self, hidden: torch.Tensor):
        logits = self.decoder.forward(hidden)
        values = self.value.forward(hidden)
        if(logits.isnan().any() or values.isnan().any()):
            pass
        return logits, values

    def unpack_obs(self, obs: torch.Tensor):
        # Header now contains only stat mods for both players (4 ints)
        # Stat mods: unpack and normalize for both players
        p1_stats = self.unpack_stats(obs[:, 0:1], obs[:, 1:2])  # [batch, 7]
        p2_stats = self.unpack_stats(obs[:, 2:3], obs[:, 3:4])  # [batch, 7]

        # Flatten into single vector: p1_stats + p2_stats (14 dims)
        active_features = torch.cat([p1_stats, p2_stats], dim=1)
        base_mat = obs[:, self.header_size:].view(-1, self.num_pokemon_rows, self.pokemon_row_length)

        # Unpack compressed tensors
        pokemon_vec, move_pp, status_vec = self.unpack_pokemon_rows(base_mat)

    # Concatenate all features: stats + pokemon data
    # active_features: [batch, 14] (p1_stats + p2_stats)
    # pokemon_vec: [batch, 12, 26] -> [batch, 312]
    # move_pp: [batch, 12, 4] -> [batch, 48]
    # status_vec: [batch, 12, 6] -> [batch, 72]
    # Total: 14 + 312 + 48 + 72 = 446
        output_matrix = torch.cat([
            active_features,
            pokemon_vec.flatten(1),
            move_pp.flatten(1),
            status_vec.flatten(1)
    ], dim=1)  # [batch, 434]

        return output_matrix
    # Embedding pokemon IDs and moves, sum per pokemon, combine with gamestate

    @staticmethod
    def unpack_stats(high, low):
        """Unpack 7 stat mods from two packed ints into separate normalized tensors (0-12)."""
        shifts_high = torch.tensor([0, 4, 8, 12], device=high.device)
        shifts_low = torch.tensor([0, 4, 8], device=low.device)
        stats_high = ((high >> shifts_high) & 0xF)  # [batch, 4]
        stats_low = ((low >> shifts_low) & 0xF)  # [batch, 3]
        stats = -1.0 + torch.cat([stats_high, stats_low], dim=1) / 6.0  
        return stats # [batch, 7]
    
    def unpack_pokemon_rows(self, base_mat: torch.Tensor):
        # Unpack species and moves
        # Species embedding
        species_ids = torch.abs(base_mat[:, :, 0]).long()
        # Map species ids into embedding range using modulo (simpler and avoids
        # overflow). This keeps values valid for embedding lookup while not
        # mutating the original packing logic.
        max_species = self.species_embed.num_embeddings
        species_emb = self.species_embed(species_ids % max_species)  # [batch, 12, 4]
        
        # Move embeddings and PP
        move_embs = []
        move_pp = []
        for k in range(4):
            move_packed = base_mat[:, :, 1 + k]
            move_ids = (move_packed & 0xFF).long()
            # Map move ids into embedding range using modulo to prevent overflow
            max_move = self.move_embed.num_embeddings
            move_emb = self.move_embed(move_ids % max_move)  # [batch, 12, embed_size]
            move_embs.append(move_emb)

            pp = ((move_packed >> 8) & 0x1F).float() / 31.0  # PP 0-31 -> 0-1
            move_pp.append(pp)

        # Active flags: species < 0 indicates active
        active_flags = (base_mat[:, :, 0] < 0).float()  # [batch, 12]


        #Batch, [12, 26]
        pokemon_vec = torch.cat([species_emb, active_flags.unsqueeze(-1)] + move_embs, dim=-1)  # [batch, 12, 26]
        move_pp = torch.stack(move_pp, dim=2)  # [batch, 12, 4]
        
        # Status: unpack individual bits (vectorized)
        status_packed = base_mat[:, :, 6]
        shifts = torch.tensor([0, 1, 2, 3, 4, 5], device=base_mat.device)
        status_vec = ((status_packed.unsqueeze(-1) >> shifts) & 1).float()  # [batch, 12, 6]
        
        return pokemon_vec, move_pp, status_vec
