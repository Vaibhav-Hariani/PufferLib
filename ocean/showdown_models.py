import torch
import torch.nn as nn


class ShowdownEncoder(nn.Module):
    # Obs layout: 14 stat-mod floats + 12 pokemon * 17 floats + 10 mask floats = 228 total
    # Per-pokemon row: [species_id/151, active, move_ids/165 *4, pp*4, hp, status*6]
    # Trailing 10 floats: legal-action mask (6 switch + 4 move), consumed here as an
    # extra input feature (it encodes state not otherwise visible, e.g. lock-in/disable)
    # and separately by ShowdownDecoder for hard logit masking.
    HEADER      = 14
    NUM_POKE    = 12
    POKE_FLOATS = 17
    MASK_FLOATS = 10
    EMBED_DIM   = 8
    MAX_SPECIES = 152   # IDs 0-151
    MAX_MOVE    = 166   # IDs 0-165

    # Concat size: 14 + 12*(species_emb + active + 4*move_emb + 4*pp + hp + 6*status) + 10
    #            = 14 + 12*(8 + 1 + 32 + 4 + 1 + 6) + 10 = 14 + 12*52 + 10 = 648
    CONCAT_SIZE = HEADER + NUM_POKE * (EMBED_DIM + 1 + 4 * EMBED_DIM + 4 + 1 + 6) + MASK_FLOATS

    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.species_embed = nn.Embedding(self.MAX_SPECIES, self.EMBED_DIM)
        self.move_embed    = nn.Embedding(self.MAX_MOVE,    self.EMBED_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(self.CONCAT_SIZE, hidden_size),
            nn.GELU(),
        )

    def forward(self, observations):
        B = observations.shape[0]
        obs = observations.float()
        header = obs[:, :self.HEADER]
        poke_end = self.HEADER + self.NUM_POKE * self.POKE_FLOATS
        rows = obs[:, self.HEADER:poke_end].view(B, self.NUM_POKE, self.POKE_FLOATS)
        mask = obs[:, poke_end:poke_end + self.MASK_FLOATS]

        # Scale normalized IDs back to integers for embedding lookup
        species_ids  = (rows[:, :, 0] * 151).long().clamp(0, self.MAX_SPECIES - 1)
        active_flags = rows[:, :, 1]
        move_ids     = (rows[:, :, 2:6] * 165).long().clamp(0, self.MAX_MOVE - 1)
        move_pp      = rows[:, :, 6:10]
        hp           = rows[:, :, 10]
        status       = rows[:, :, 11:17]

        species_emb = self.species_embed(species_ids)   # [B, 12, 8]
        move_emb    = self.move_embed(move_ids)          # [B, 12, 4, 8]

        x = torch.cat([
            header,
            species_emb.flatten(1),
            active_flags,
            move_emb.flatten(1),
            move_pp.flatten(1),
            hp,
            status.flatten(1),
            mask,
        ], dim=1)
        return self.encoder(x)


class ShowdownDecoder(nn.Module):
    # Mirrors pufferlib.models.DefaultDecoder's interface/behavior for a single
    # Discrete(10) action head, but additionally hard-masks illegal actions
    # using the trailing MASK_FLOATS of the raw observation (see ShowdownEncoder).
    MASK_FLOATS = ShowdownEncoder.MASK_FLOATS

    def __init__(self, nvec, hidden_size=128):
        super().__init__()
        nvec = tuple(nvec)
        assert sum(nvec) != len(nvec), "ShowdownDecoder expects a discrete action space"
        self.decoder = nn.Linear(hidden_size, int(sum(nvec)))
        self.value_function = nn.Linear(hidden_size, 1)

    def forward(self, hidden, obs=None):
        logits = self.decoder(hidden)
        if obs is not None:
            mask = obs[:, -self.MASK_FLOATS:] > 0.5
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        values = self.value_function(hidden)
        return logits, values
