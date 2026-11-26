"""Tests for guppylm/stages.py. Run from the repo root:  python -m unittest discover -s tests -v"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from guppylm.config import GuppyConfig
from guppylm.inference import GuppyInference
from guppylm.model import GuppyLM
from guppylm.stages import (
    PackedDataset, get_pretrain_loader, lm_loss, train_stage, write_pretrain_corpus,
)

SENTENCES = [f"the water is {w} today" for w in ["warm", "cold", "clear", "cloudy", "green", "still"]] * 40 + \
            [f"i see a {w}" for w in ["bubble", "snail", "plant", "rock", "log", "flake"]] * 40


def make_tokenizer(path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=200, special_tokens=["<pad>", "<|im_start|>", "<|im_end|>"], min_frequency=1)
    tok.train_from_iterator(SENTENCES, trainer)
    tok.save(path)


class StagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.tok_path = os.path.join(d, "tokenizer.json")
        cls.train_path = os.path.join(d, "train.jsonl")
        cls.eval_path = os.path.join(d, "eval.jsonl")
        make_tokenizer(cls.tok_path)
        cls.counts = write_pretrain_corpus(SENTENCES, cls.train_path, cls.eval_path, eval_fraction=0.25)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def read(self, path):
        with open(path) as f:
            return [json.loads(line)["text"] for line in f]

    def test_corpus_is_deduplicated_and_split_without_overlap(self):
        train, ev = self.read(self.train_path), self.read(self.eval_path)
        self.assertEqual(len(set(SENTENCES)), 12)                 # 12 unique sentences in the toy data
        self.assertEqual(len(train) + len(ev), 12)
        self.assertEqual(len(set(train)), len(train))             # no duplicates inside train
        self.assertFalse(set(train) & set(ev))                    # no sentence in both splits
        self.assertEqual(self.counts, (len(train), len(ev)))

    def test_corpus_is_repeatable(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
            write_pretrain_corpus(SENTENCES, a, b, eval_fraction=0.25, seed=42)
            self.assertEqual(self.read(a), self.read(self.train_path))

    def test_packed_blocks_are_shifted_by_one_and_contiguous(self):
        ds = PackedDataset(self.train_path, self.tok_path, block_size=8, end_id=2)
        self.assertGreater(len(ds), 1)
        for i in range(len(ds)):
            x, y = ds[i]
            self.assertEqual(x.shape, (8,))
            self.assertEqual(y.shape, (8,))
            self.assertTrue(torch.equal(x[1:], y[:-1]))           # y is x shifted left by one
            self.assertTrue(torch.equal(x, ds.tokens[i * 8:(i + 1) * 8]))
        x1, _ = ds[1]
        _, y0 = ds[0]
        self.assertEqual(y0[-1].item(), x1[0].item())             # block 1 starts right where block 0's target ends

    def test_every_text_ends_with_the_end_token(self):
        ds = PackedDataset(self.train_path, self.tok_path, block_size=8, end_id=2)
        n_texts = len(self.read(self.train_path))
        self.assertEqual((ds.tokens == 2).sum().item(), n_texts)  # one end token per text, no more

    def test_pretraining_lowers_loss_and_checkpoint_loads_in_inference(self):
        torch.manual_seed(0)
        cfg = GuppyConfig(vocab_size=200, max_seq_len=8, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, dropout=0.0)
        model = GuppyLM(cfg)
        train_loader = get_pretrain_loader(self.train_path, self.tok_path, 8, batch_size=2)
        eval_loader = get_pretrain_loader(self.eval_path, self.tok_path, 8, batch_size=2, shuffle=False)
        with torch.no_grad():
            first = lm_loss(model, next(iter(train_loader))).item()
        save_path = os.path.join(self.dir.name, "ckpt", "pretrain.pt")
        losses, best = train_stage(model, train_loader, eval_loader, lm_loss, torch.device("cpu"), save_path,
                                   max_steps=60, learning_rate=3e-3, warmup_steps=5, eval_interval=20, log_every=20)
        self.assertGreater(first, 4.0)                            # near ln(200) = 5.3 before training
        self.assertLess(sum(losses[-10:]) / 10, first - 1.0)      # clearly lower after training
        self.assertTrue(os.path.exists(save_path))
        engine = GuppyInference(save_path, self.tok_path, device="cpu")   # the saved file loads like any checkpoint
        self.assertEqual(engine.config.d_model, 32)

    def test_empty_train_loader_raises_instead_of_hanging(self):
        cfg = GuppyConfig(vocab_size=200, max_seq_len=8, d_model=32, n_layers=1, n_heads=2, ffn_hidden=64)
        too_big = get_pretrain_loader(self.train_path, self.tok_path, 8, batch_size=1000)   # more than the data holds
        with self.assertRaises(ValueError):
            train_stage(GuppyLM(cfg), too_big, too_big, lm_loss, torch.device("cpu"),
                        os.path.join(self.dir.name, "never.pt"), max_steps=5)


if __name__ == "__main__":
    unittest.main()
