import torch
import torch.nn as nn

from bellm.tokeniser import Tokeniser
from reader import get_dataset


class TextDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size=10000, context_len=1000, diffuse_len=200, model_dim=512):
        super().__init__()

        self.diffuse_len = diffuse_len

        self.model_dim = model_dim

        # 1. Context Embedding (for the 1k fixed tokens)
        self.context_embedding = nn.Embedding(vocab_size, model_dim)

        # 2. Logit Projection (for the 200 tokens being diffused)
        # This takes the B x 200 x 10000 logits and brings them to model_dim
        self.diffuse_projection = nn.Linear(vocab_size, model_dim)

        # 3. Time Embedding (MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )

        # 4. Transformer Layers
        # We use a standard Encoder; you can increase num_layers for better results
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=model_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 5. Output Head (Back to 10k logits)
        self.to_logits = nn.Linear(model_dim, vocab_size)

    def forward(self, context_ids, noisy_logits, t):
        """
        context_ids: (B, 1000) - Discrete token IDs
        noisy_logits: (B, 200, 10000) - Continuous logit space
        t: (B, 1) - Timestep [0, 1]
        """
        # Embed context
        c_emb = self.context_embedding(context_ids)  # (B, 1000, dim)

        # Project noisy logits
        d_emb = self.diffuse_projection(noisy_logits)  # (B, 200, dim)

        # Combine sequences (Total length: 1200)
        x = torch.cat([c_emb, d_emb], dim=1)

        # Inject Time
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B, 1, dim)
        x = x + t_emb

        # Process through Transformer
        x = self.transformer(x)

        # Slice out only the 200 "diffuse" tokens for the output
        diffuse_out = x[:, -self.diffuse_len:, :]

        return self.to_logits(diffuse_out)  # (B, 200, 10000)


# --- Test Run ---
model = TextDiffusionTransformer()
B = 4
ctx = torch.randint(0, 10000, (B, 1000))
logits = torch.randn(B, 200, 10000)
time = torch.rand(B, 1)

output = model(ctx, logits, time)
print(f"Output shape: {output.shape}")  # torch.Size([4, 200, 10000])


if __name__ == "__main__":
    dataset = get_dataset()
    tokeniser = Tokeniser().load(f"../tokeniser/tokeniser.json")

    items = list(dataset \
         # .skip(i * SAMPLE_SIZE_BATCH_SIZE + start_offset) \
         .take(256)["text"])

    text_tokens = tokeniser.tokenize_batch(items)
    for item in text_tokens:
        tokens = item.token_ids[:1200]
        train, test = item[:1000], item[1000:]

        output = model(train)