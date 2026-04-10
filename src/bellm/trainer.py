import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from bellm.tokeniser import Tokeniser
from reader import get_dataset

DEVICE = torch.device("cuda")
EPOCHS = 50
BATCH_SIZE = 25
MAX_VOCAB = 5000
INPUT_CONTEXT_SIZE = 600
OUTPUT_CONTEXT_SIZE = 100

DATASET_SAMPLES = 2_500_000


def tensor(x):
    return torch.tensor(x, dtype=torch.long, device=DEVICE)


dataset = get_dataset()
tokeniser = Tokeniser().load(f"tokeniser/tokeniser.json")
for x in list(tokeniser.token_map):
    if tokeniser.token_map[x] >= MAX_VOCAB:
        del tokeniser.token_map[x]

print("Loading dataset (todo cache)")
items = list(dataset.take(DATASET_SAMPLES)["text"])

print("Tokenising dataset (todo cache)")
all_items = []
batch_size = 5000
for batch in tqdm(range(0, len(items), batch_size)):
    all_items += [
        tokeniser.tokenize(x).token_ids
        for x in items[batch:batch + batch_size]
    ]


split = len(all_items) // 10
train_set = all_items[split:]
val_set = all_items[:split]


import math


class TextDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size=5000, context_len=600, diffuse_len=200, model_dim=1024):
        super().__init__()

        self.vocab_size = vocab_size
        self.diffuse_len = diffuse_len
        self.model_dim = model_dim

        # 1. Context Embedding (for the 1k fixed tokens)
        self.context_embedding = nn.Embedding(vocab_size, model_dim)

        # 2. Logit Projection (for the 200 tokens being diffused)
        # This takes the B x 200 x 10000 logits and brings them to model_dim
        self.diffuse_projection = nn.Linear(vocab_size, model_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, context_len + diffuse_len, model_dim))

        # IMPROVEMENT: Use a more expressive time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim)
        )

        # 4. Transformer Layers
        # We use a standard Encoder; you can increase num_layers for better results
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=16,
            dim_feedforward=model_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=7)

        # 5. Output Head (Back to 10k logits)
        self.to_velocity = nn.Linear(model_dim, vocab_size)

        self.input_norm = nn.LayerNorm(model_dim)

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


model = TextDiffusionTransformer(
    vocab_size=MAX_VOCAB,
    context_len=INPUT_CONTEXT_SIZE,
    diffuse_len=OUTPUT_CONTEXT_SIZE
)

model.to(DEVICE).to(torch.bfloat16)
print("Model Size:", sum(p.numel() for p in model.parameters()))

import random


def preprocess_batch(batch):
    X, Y = [], []
    for tokens in batch:
        t_idx = random.randint(0, len(tokens) - 1)

        input = tokens[:t_idx]
        output = tokens[t_idx + 1:]

        if len(input) < INPUT_CONTEXT_SIZE:
            input = [Tokeniser.PAD] * (INPUT_CONTEXT_SIZE - len(input)) + input

        if len(output) < OUTPUT_CONTEXT_SIZE:
            output = output + [Tokeniser.PAD] * (OUTPUT_CONTEXT_SIZE - len(output))

        if len(input) > INPUT_CONTEXT_SIZE:
            input = input[-INPUT_CONTEXT_SIZE:]

        # If the output is longer than context size, then truncate it and at a next page token
        if len(output) > OUTPUT_CONTEXT_SIZE:
            output = output[:OUTPUT_CONTEXT_SIZE - 1] + [Tokeniser.NEXT_PAGE]

        X.append(input)
        Y.append(output)

    return (
        torch.tensor(np.array(X), dtype=torch.long, device=DEVICE),
        torch.tensor(np.array(Y), dtype=torch.long, device=DEVICE)
    )

optimiser = torch.optim.Adam( model.parameters(),
                                 lr = 0.001,
                                 weight_decay = 0.002)

my_lr_scheduler = torch.optim.lr_scheduler.StepLR( optimiser,
                                                step_size = 1,
                                                gamma = 0.9)

for epoch in range(EPOCHS):
    train_losses = []
    val_losses = []

    model.train()
    for b_idx in range(0, len(train_set), BATCH_SIZE):
        batch_x, batch_y = preprocess_batch(train_set[b_idx:b_idx + BATCH_SIZE])

        optimiser.zero_grad(set_to_none=True)

        # 1. Create Clean State (One-Hot)
        # Bfloat16 saves memory and is usually enough for these models
        x0 = F.one_hot(batch_y, num_classes=MAX_VOCAB).to(DEVICE).to(torch.bfloat16)

        # 2. Create Noise State (Uniform)
        x1 = torch.full_like(x0, 1.0 / MAX_VOCAB)

        # 3. Sample Time
        t = torch.rand(len(batch_x), 1, device=DEVICE).to(torch.bfloat16)
        # Reshape t to (B, 1, 1) for broadcasting across (B, 200, 10000)
        t_view = t.view(-1, 1, 1)

        # 4. Construct Noisy Mixture
        xt = (1 - t_view) * x0 + t_view * x1

        # 5. Forward Pass
        # We pass batch_x (context), xt (noisy targets), and t (time)
        pred_logits = model(batch_x, xt, t)

        # 6. Loss Calculation
        # Target is the original token IDs (batch_y)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, MAX_VOCAB),
            batch_y.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimiser.step()

        train_losses.append(loss.item())
        if b_idx % 100 == 0:
            print(f"\rEpoch: {epoch} | Batch: {b_idx} | Loss: {np.mean(train_losses):.4f} lr={my_lr_scheduler.get_lr()}", end="")


    model.eval()
    for b_idx in range(0, len(val_set), BATCH_SIZE):
        batch_x, batch_y = preprocess_batch(val_set[b_idx:b_idx + BATCH_SIZE])

        x0 = F.one_hot(batch_y, num_classes=MAX_VOCAB).to(DEVICE).to(torch.bfloat16)
        x1 = torch.full_like(x0, 1.0 / MAX_VOCAB)

        t = torch.rand(len(batch_x), 1, device=DEVICE).to(torch.bfloat16)
        t_view = t.view(-1, 1, 1)

        xt = (1 - t_view) * x0 + t_view * x1

        pred_logits = model(batch_x, xt, t)

        loss = F.cross_entropy(
            pred_logits.reshape(-1, MAX_VOCAB),
            batch_y.reshape(-1)
        )

        val_losses.append(loss.item())
        if b_idx % 100 == 0:
            print(f"\rEpoch: {epoch} (Val) | Batch: {b_idx} | Loss: {np.mean(val_losses):.4f} lr={my_lr_scheduler.get_lr()}", end="")

    torch.save(model.state_dict(), "model.pt")
    torch.save(optimiser.state_dict(), "optim.pt")
    my_lr_scheduler.step()

    print(f"\r--- Epoch {epoch} Complete. Avg Loss: {np.mean(train_losses):.4f}  Val Avg Loss: {np.mean(val_losses):.4f} lr={my_lr_scheduler.get_lr()} ---")


# TESTING


model.eval()

text = "is"
tokens = tokeniser.tokenize_batch([text])[0].token_ids
tokens = [Tokeniser.PAD] * (INPUT_CONTEXT_SIZE - len(tokens)) + tokens
model_input = torch.tensor(np.array([tokens]), dtype=torch.long, device=DEVICE)


def generate_flow(model, context_ids, steps=20):
    B = context_ids.shape[0]
    # Start at t=1 (Pure Noise)
    xt = torch.randn(B, OUTPUT_CONTEXT_SIZE, MAX_VOCAB, dtype=torch.bfloat16).to(context_ids.device)

    dt = 1.0 / steps

    for i in range(steps):
        # Current time (1.0 down to 0.0)
        t_val = 1.0 - (i * dt)
        t = torch.full((B, 1), t_val, dtype=torch.bfloat16).to(context_ids.device)

        # Predict velocity
        v = model(context_ids, xt, t)

        # Euler Step: Update xt by moving in the direction of -velocity
        # (Minus because we are going from noise -> data)
        xt = xt - v * dt

    return xt  # These are your denoised pre-softmax logits


softmax = F.softmax(generate_flow(model, model_input), dim=2)
tokens = torch.argmax(softmax, dim=2).detach().cpu().numpy()

print("".join(tokeniser.detokenise(tokens[0])))