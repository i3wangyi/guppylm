"""GuppyLM dataset loading."""

import json

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


# A PyTorch Dataset answers two questions: "how many samples?" (__len__) and
# "give me sample number i" (__getitem__). The DataLoader below builds batches from it.
class GuppyDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []

        # Read the JSONL file (one JSON object per line) and turn every text into token ids once,
        # up front. All 57,000 samples stay in memory, which is fine because they are short.
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                ids = self.tokenizer.encode(data["text"]).ids
                if len(ids) > max_len:
                    ids = ids[:max_len]        # cut anything longer than the model can read
                if len(ids) >= 2:              # need at least one input token and one target token
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # The whole training task is here. The target is the input shifted by one token:
        #   input  x: [A, B, C, D]
        #   target y: [B, C, D, E]
        # At each position the model must predict the NEXT token.
        x = ids[:-1]
        y = ids[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    # Sentences have different lengths, but a tensor must be a rectangle.
    # So we pad every sentence in the batch with pad_id up to the longest one.
    # The loss ignores pad_id (see ignore_index=0 in model.py), so padding does not teach anything.
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    return padded_x, padded_y


def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True):
    dataset = GuppyDataset(path, tokenizer_path, max_len)
    # shuffle=True gives a new random order each epoch (one pass over the data) for training.
    # num_workers=0 loads batches in the main process: simple, and fast enough for tiny data.
    # pin_memory=True puts batches in page-locked memory, which makes copying to the GPU faster.
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
