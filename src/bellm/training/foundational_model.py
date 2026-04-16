import os
from pathlib import Path

from bellm.cli.training import FoundationModelTrainingConfig
from bellm.dataloader.foundation_model_dataloader import FoundationDataLoader
from bellm.logging.tensorboard import MLflowInterface
from bellm.model.bellm_v1 import TextDiffusionTransformer
from bellm.utils import get_device

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F
import numpy as np

from bellm.tokeniser import Tokeniser

DEVICE = get_device()

def train_foundational_model(args: FoundationModelTrainingConfig):
    LOG_EVERY_BATCH_ITER = 10

    tokeniser = Tokeniser().load(args.model.tokeniser)
    MAX_VOCAB = len(tokeniser)

    model = TextDiffusionTransformer(
        vocab_size=MAX_VOCAB,
        context_len=args.model.input_context_size,
        diffuse_len=args.model.output_context_size,
        embedding_dim=args.model.embedding_dim,
        transformer_layers=args.model.layers,
        n_heads=args.model.heads,
        sliding_window_size=args.model.window
    )

    model.to(DEVICE).to(torch.bfloat16)
    print("Model Size:", sum(p.numel() for p in model.parameters()))

    logger = MLflowInterface(experiment_name="bellm-v1", run_name="test1")

    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.002)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.9)

    training_dataset = FoundationDataLoader(
        Path("/data/preprocessed/foundation/train"),
        batch_size=args.batch_size,
        tokeniser=tokeniser,
        input_context_length=args.model.input_context_size,
        output_context_length=args.model.output_context_size,
    )

    validation_dataset = FoundationDataLoader(
        Path("/data/preprocessed/foundation/validation"),
        batch_size=args.batch_size,
        tokeniser=tokeniser,
        input_context_length=args.model.input_context_size,
        output_context_length=args.model.output_context_size,
    )

    if __name__ == "__main__":
        for epoch in range(1, args.epochs + 1):
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
                if bidx % LOG_EVERY_BATCH_ITER == 0:
                    logger.log_training_epoch_data(epoch, bidx, training_dataset.batch_count, np.mean(train_losses))

            model.eval()
            for bidx, (batch_x, batch_y) in enumerate(validation_dataset):
                batch_x = torch.tensor(batch_x, dtype=torch.long, device=DEVICE)
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
                if bidx % LOG_EVERY_BATCH_ITER == 0:
                    logger.log_validation_epoch_data(epoch, bidx, validation_dataset.batch_count, np.mean(val_losses))

            logger.log_epoch_data(
                np.mean(train_losses),
                np.mean(val_losses),
                lr_scheduler.get_lr()[0],
                epoch=epoch
            )

            torch.save(model.state_dict(), "model.pt")
            torch.save(optimiser.state_dict(), "optim.pt")
            lr_scheduler.step()

            # todo add test examples for text output

            print(f"\r--- Epoch {epoch} Complete. Avg Loss: {np.mean(train_losses):.4f}  Val Avg Loss: {np.mean(val_losses):.4f} ---")

        # TESTING

            model.eval()

            def generate_flow(text, model, steps=20):
                tokens = tokeniser.tokenize_batch([text])[0].token_ids
                tokens = tokens + [Tokeniser.PAD] * (args.model.input_context_size - len(tokens))
                context_ids = torch.tensor(np.array([tokens]), dtype=torch.long, device=DEVICE)

                B = context_ids.shape[0]
                # Start at t=1 (Pure Noise)
                xt = torch.randn(B, args.model.output_context_size, MAX_VOCAB, dtype=torch.bfloat16).to(context_ids.device)

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

                softmax = F.softmax(xt, dim=2)
                tokens = torch.argmax(softmax, dim=2).detach().cpu().numpy()

                return "".join(tokeniser.detokenise(tokens[0]))

            # Log example of text
            prompts = [
                "Hello, how are you?",
                "Why are you gey?",
                "How do i make a casserole?",
                "Please explain rummikub to me?",
                "Whats the best opening move in chess?",
            ]
            prompts = [(text_input, "[USER]: " + text_input + "\n[ASSISTANT]: ") for text_input in prompts]

            logger.log_test_text(
                [
                    (
                        title,
                        message + generate_flow(message, model),
                    )
                    for title, message in prompts
                ],
                epoch=epoch
            )
