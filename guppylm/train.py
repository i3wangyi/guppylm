"""GuppyLM training loop."""

import json
import math
import os
import time

import torch

from .config import GuppyConfig, TrainConfig
from .dataset import get_dataloader
from .model import GuppyLM


def get_device(config):
    # Pick where the math runs. A GPU is much faster than a CPU for this kind of work.
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")      # NVIDIA GPU (Colab)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")       # Apple Silicon GPU
        return torch.device("cpu")
    return torch.device(config.device)


def get_lr(step, config):
    # Learning rate schedule: linear warmup, then cosine decay.
    # Warmup: at the start the weights are random and the gradients are noisy, so take small steps first.
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    # progress runs from 0.0 (end of warmup) to 1.0 (last step).
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    # coeff follows half a cosine wave: 1.0 at progress 0, falling smoothly to 0.0 at progress 1.
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    # Blend between the final rate (min_lr) and the peak rate (learning_rate) using coeff.
    return config.min_lr + (config.learning_rate - config.min_lr) * coeff


@torch.no_grad()  # no gradients needed when we only measure the loss
def evaluate(model, loader, device, max_batches=50):
    # Average loss on data the model has NOT trained on. If this stops improving while the training
    # loss keeps falling, the model is memorising instead of learning.
    model.eval()   # turns dropout off, so results are not random
    total_loss, n = 0, 0
    for x, y in loader:
        if n >= max_batches:   # 50 batches is enough for a stable estimate, and it keeps evaluation quick
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()   # .item() turns a one-number tensor into a normal Python float
        n += 1
    model.train()  # back to training mode (dropout on)
    return total_loss / max(1, n)


def train():
    mc = GuppyConfig()   # model settings
    tc = TrainConfig()   # training settings
    device = get_device(tc)
    torch.manual_seed(tc.seed)   # same seed -> same random start -> repeatable runs

    print(f"Device: {device}")

    tokenizer_path = os.path.join(tc.data_dir, "tokenizer.json")
    model = GuppyLM(mc).to(device)   # .to(device) copies all weights onto the GPU
    print(model.param_summary())

    # Training data is shuffled every epoch. Eval data is not, so eval numbers are comparable between runs.
    train_loader = get_dataloader(
        os.path.join(tc.data_dir, "train.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=True,
    )
    eval_loader = get_dataloader(
        os.path.join(tc.data_dir, "eval.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=False,
    )
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    # The optimizer decides how to change each weight, given its gradient. AdamW is the standard choice
    # for transformers: it adapts the step size per weight and applies weight decay properly.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate,
        weight_decay=tc.weight_decay, betas=(0.9, 0.95),
    )

    # Mixed precision: on a GPU, run most math in 16-bit numbers. It is faster and uses less memory.
    # 16-bit gradients can be so small that they round to zero, so GradScaler multiplies the loss by a
    # large number before the backward pass and divides it back out afterwards.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    os.makedirs(tc.output_dir, exist_ok=True)
    # Save the settings next to the weights, so inference can rebuild a model of the same shape.
    with open(os.path.join(tc.output_dir, "config.json"), "w") as f:
        json.dump({"model": vars(mc), "train": vars(tc)}, f, indent=2)

    model.train()   # turn dropout on
    step, best_eval = 0, float("inf")
    losses = []     # the loss of every step, kept so we can plot it later
    t0 = time.time()

    print(f"\nTraining for {tc.max_steps} steps...")
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    # One "step" = one batch = one weight update. The outer loop restarts the data when it runs out
    # (a new epoch) and the inner break stops exactly at max_steps.
    while step < tc.max_steps:
        for x, y in train_loader:
            if step >= tc.max_steps:
                break

            x, y = x.to(device), y.to(device)
            # Set this step's learning rate from the schedule.
            lr = get_lr(step, tc)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if use_amp:
                # GPU path, with mixed precision.
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)                    # 1. forward pass: predictions and loss
                scaler.scale(loss).backward()                # 2. backward pass: a gradient for every weight
                scaler.unscale_(optimizer)                   # undo the loss scaling before we look at the gradients
                # Gradient clipping: if the total gradient is larger than grad_clip, shrink it to that size.
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(optimizer)                       # 3. update the weights (skipped if gradients overflowed)
                scaler.update()                              # adjust the scale factor for the next step
            else:
                # CPU / Apple path, plain 32-bit.
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()

            # 4. Reset gradients. PyTorch ADDS new gradients to old ones, so we must clear them each step.
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % 100 == 0:
                # Print the average of the last 100 losses, which is steadier than a single noisy value.
                avg = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {elapsed:7.1f}s")

            if step > 0 and step % tc.eval_interval == 0:
                el = evaluate(model, eval_loader, device)
                avg_train = sum(losses[-tc.eval_interval:]) / min(len(losses), tc.eval_interval)
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg_train:10.4f} | {el:10.4f} | {elapsed:7.1f}s")

                # Keep the checkpoint with the lowest eval loss. This guards against a later, worse step.
                if el < best_eval:
                    best_eval = el
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),   # all learned weights, as a dict of tensors
                        "config": vars(mc),
                        "eval_loss": el,
                    }, os.path.join(tc.output_dir, "best_model.pt"))
                    print(f"  -> Best model (eval={el:.4f})")

            # Periodic snapshots, so a crashed run can resume from the latest one.
            if step > 0 and step % tc.save_interval == 0:
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": vars(mc),
                }, os.path.join(tc.output_dir, f"step_{step}.pt"))

            step += 1

    # Final save
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": vars(mc),
        "train_losses": losses,
    }, os.path.join(tc.output_dir, "final_model.pt"))

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.0f}s, best eval: {best_eval:.4f}")


if __name__ == "__main__":
    train()
