import math

import torch
import torch.nn as nn


class TextDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size=5000, context_len=600, diffuse_len=200, embedding_dim=512, transformer_layers=12):
        super().__init__()

        self.vocab_size = vocab_size
        self.diffuse_len = diffuse_len
        self.model_dim = embedding_dim

        # 1. Context Embedding (for the 1k fixed tokens)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Logit Projection (for the 200 tokens being diffused)
        # This takes the B x 200 x 10000 logits and brings them to model_dim
        self.diffuse_projection = nn.Linear(vocab_size, embedding_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, context_len + diffuse_len, embedding_dim))

        # IMPROVEMENT: Use a more expressive time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        # 4. Transformer Layers
        # We use a standard Encoder; you can increase num_layers for better results
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=16,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 5. Output Head (Back to 10k logits)
        self.to_velocity = nn.Linear(embedding_dim, vocab_size)

        self.input_norm = nn.LayerNorm(embedding_dim)

    def timestep_embedding(self, timesteps, dim):
        half_dim = dim // 2
        # Ensure 'exponent' is the same dtype as timesteps (bfloat16)
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device).to(timesteps.dtype) / half_dim
        freqs = torch.exp(exponent)

        args = timesteps * freqs
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, context_ids, noisy_logits, t):
        """

        noisy_logits: (B, 200, 10000) - Continuous logit space
        t: (B, 1) - Timestep [0, 1]
        """

        # Embed context
        c_emb = self.context_embedding(context_ids)  # (B, 1000, dim)

        # Project noisy logits
        d_emb = self.diffuse_projection(noisy_logits)  # (B, 200, dim)
        d_emb = self.input_norm(d_emb)

        # Combine sequences (Total length: 1200)
        x = torch.cat([c_emb, d_emb], dim=1)

        x = x + self.pos_emb

        # Time injection
        t_freq = self.timestep_embedding(t, self.model_dim)  # (B, dim)
        t_emb = self.time_mlp(t_freq).unsqueeze(1)  # (B, 1, dim)
        x = x + t_emb

        # Process through Transformer
        x = self.transformer(x)

        # Slice out only the 200 "diffuse" tokens for the output
        diffuse_out = x[:, -self.diffuse_len:, :]

        pred_x0 = self.to_velocity(diffuse_out)
        # pred_x0 = F.softmax(pred_x0, dim=-1)

        # The velocity on the simplex is the straight line:
        # v = (pred_x0 - noise) / (1 - t)
        return pred_x0

