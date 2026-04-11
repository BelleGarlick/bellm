import os
from pathlib import Path

from bellm.dataloader.foundation_model_dataloader import FoundationDataLoader
from bellm.model.bellm_v1 import TextDiffusionTransformer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
import numpy as np

from bellm.tokeniser import Tokeniser

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
EPOCHS = 50
BATCH_SIZE = 25
MAX_VOCAB = 5000
INPUT_CONTEXT_SIZE = 600
OUTPUT_CONTEXT_SIZE = 100

DATASET_SAMPLES = 2_500_000


def tensor(x):
    return torch.tensor(x, dtype=torch.long, device=DEVICE)


tokeniser = Tokeniser().load(f"tokeniser/tokeniser.json")
for x in list(tokeniser.token_map):
    if tokeniser.token_map[x] >= MAX_VOCAB:
        del tokeniser.token_map[x]


model = TextDiffusionTransformer(
    vocab_size=MAX_VOCAB,
    context_len=INPUT_CONTEXT_SIZE,
    diffuse_len=OUTPUT_CONTEXT_SIZE,
    embedding_dim=512,
    transformer_layers=3
)

model.to(DEVICE).to(torch.bfloat16)
print("Model Size:", sum(p.numel() for p in model.parameters()))


optimiser = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.002
)

my_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimiser,
    step_size=1,
    gamma=0.9
)

training_dataset = FoundationDataLoader(
    Path("/Users/belle/Developer/Belllm/belllm/data/preprocessed/foundation/train"),
    batch_size=BATCH_SIZE,
    tokeniser=tokeniser,
    input_context_length=INPUT_CONTEXT_SIZE,
    output_context_length=OUTPUT_CONTEXT_SIZE,
)

validation_dataset = FoundationDataLoader(
    Path("/Users/belle/Developer/Belllm/belllm/data/preprocessed/foundation/validation"),
    batch_size=BATCH_SIZE,
    tokeniser=tokeniser,
    input_context_length=INPUT_CONTEXT_SIZE,
    output_context_length=OUTPUT_CONTEXT_SIZE,
)

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        train_losses = []
        val_losses = []

        model.train()

        for bidx, (batch_x, batch_y) in enumerate(training_dataset):
            batch_x = torch.tensor(batch_x, dtype=torch.long, device=DEVICE)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)

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
            # if bidx % 1 == 0:
            print(f"\rEpoch: {epoch} | Batch: {bidx} | Loss: {np.mean(train_losses):.4f} lr={my_lr_scheduler.get_lr()}", end="")


        model.eval()
        for bidx, (batch_x, batch_y) in enumerate(validation_dataset):
            batch_x = torch.tensor(batch_x, dtype=torch.long, device=DEVICE),
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)

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
            if bidx % 100 == 0:
                print(f"\rEpoch: {epoch} (Val) | Batch: {bidx} | Loss: {np.mean(val_losses):.4f} lr={my_lr_scheduler.get_lr()}", end="")

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
