"""Training stages for GuppyLM: pretraining first, then SFT and DPO.

Every stage uses the same model and the same training loop (train_stage). What changes from
stage to stage is the DATA and the LOSS FUNCTION:

  Stage 0  pretraining  plain text, predict the next token everywhere      (this file, below)
  Stage 1  SFT          chat text, predict the next token in answers only  (added later)
  Stage 2  DPO          pairs of good and bad answers                      (added later)
"""

import json
import os
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from .config import TrainConfig
from .train import get_lr


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 0: PRETRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════

def write_pretrain_corpus(sentences, train_path, eval_path, eval_fraction=0.02, seed=42):
    """Write a plain-text corpus: one sentence per JSONL line, no chat format.

    Steps, like a real pretraining pipeline in miniature:
      1. remove exact duplicates (repeated text wastes training and leaks into the eval set),
      2. shuffle, so neighbouring sentences are unrelated,
      3. hold out a small part for evaluation.
    """
    unique = sorted(set(s.strip() for s in sentences if s.strip()))  # sorted() first: set order is not stable
    random.Random(seed).shuffle(unique)                              # a fixed seed makes the result repeatable
    n_eval = max(1, int(len(unique) * eval_fraction))
    splits = {eval_path: unique[:n_eval], train_path: unique[n_eval:]}
    for path, items in splits.items():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for s in items:
                f.write(json.dumps({"text": s}) + "\n")
    return len(unique) - n_eval, n_eval


class PackedDataset(Dataset):
    """Glue all texts into one long stream of tokens, then cut it into equal blocks.

    Real pretraining does the same. It wastes no space on padding, and every block is full.
    Texts are separated by the end-of-text token, so the model learns where a text ends.

      stream:  [the, water, is, warm, <end>, i, love, food, <end>, ...]
      block 0: x = stream[0:128]    y = stream[1:129]    (y is x shifted by one token)
      block 1: x = stream[128:256]  y = stream[129:257]
    """

    def __init__(self, path, tokenizer_path, block_size=128, end_id=2):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        ids = []
        with open(path) as f:
            for line in f:
                ids.extend(tokenizer.encode(json.loads(line)["text"]).ids)
                ids.append(end_id)
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        # Each block needs block_size + 1 tokens (the extra one is the last target). Leftover tokens are dropped.
        self.n_blocks = (len(ids) - 1) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, i):
        start = i * self.block_size
        chunk = self.tokens[start:start + self.block_size + 1]
        return chunk[:-1], chunk[1:]   # all blocks have the same length, so no padding or custom collate is needed


def get_pretrain_loader(path, tokenizer_path, block_size=128, batch_size=32, shuffle=True, end_id=2):
    dataset = PackedDataset(path, tokenizer_path, block_size, end_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS. A loss function takes (model, batch) and returns one number.
# ══════════════════════════════════════════════════════════════════════════════

def lm_loss(model, batch):
    """Plain next-token loss: the model's own cross-entropy, counted at every position."""
    x, y = batch
    _, loss = model(x, y)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
#  THE TRAINING LOOP, SHARED BY ALL STAGES
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_stage(model, loader, loss_fn, device, max_batches=50):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        if n >= max_batches:
            break
        total += loss_fn(model, [t.to(device) for t in batch]).item()
        n += 1
    model.train()
    return total / max(1, n)


def train_stage(model, train_loader, eval_loader, loss_fn, device, save_path,
                max_steps=2000, learning_rate=3e-4, min_lr=3e-5, warmup_steps=100,
                eval_interval=200, log_every=100, weight_decay=0.1, grad_clip=1.0):
    """Train `model` with `loss_fn`. Keep the weights with the lowest eval loss at `save_path`.

    This is the same loop as train.py (forward, backward, clip, update, reset), with the loss
    pulled out into `loss_fn` so that other stages can reuse it. Returns (step_losses, best_eval).
    """
    if len(train_loader) == 0:
        # Without this check, the while loop below would spin forever on an empty loader.
        raise ValueError("train_loader has no batches. Use a smaller batch_size or block_size, or more data.")
    # TrainConfig is reused only as a bag of numbers for get_lr (warmup, cosine decay).
    tc = TrainConfig(max_steps=max_steps, learning_rate=learning_rate, min_lr=min_lr, warmup_steps=warmup_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
    use_amp = device.type == "cuda"                              # mixed precision, as in train.py
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    model.train()
    step, best_eval, losses, t0 = 0, float("inf"), [], time.time()
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            batch = [t.to(device) for t in batch]
            lr = get_lr(step, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = loss_fn(model, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if step % log_every == 0:
                avg = sum(losses[-log_every:]) / len(losses[-log_every:])
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {time.time() - t0:7.1f}s")

            if step > 0 and step % eval_interval == 0:
                eval_loss = evaluate_stage(model, eval_loader, loss_fn, device)
                avg = sum(losses[-eval_interval:]) / min(len(losses), eval_interval)
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {eval_loss:10.4f} | {time.time() - t0:7.1f}s")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": vars(model.config),
                        "eval_loss": eval_loss,
                    }, save_path)
                    print(f"  -> Best model (eval={eval_loss:.4f})")
            step += 1

    print(f"\nDone! {time.time() - t0:.0f}s, best eval: {best_eval:.4f}")
    return losses, best_eval
