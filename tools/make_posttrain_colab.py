"""Generate the post-training Colab notebook (pretraining -> SFT -> DPO).

Run:  python3 tools/make_posttrain_colab.py
Output: guppylm_post_training.ipynb (repo root)

Like make_beginner_colab.py, the notebook writes the real source files from guppylm/ into Colab.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_colab import PROJECT_ROOT, code, md, read_for_colab, write_notebook  # noqa: E402


def src(name):
    return read_for_colab(os.path.join(PROJECT_ROOT, "guppylm", name)).strip("\n")


def c(text):
    return code(text.strip("\n"))


def m(text):
    return md(text.strip("\n"))


def build():
    cells = []

    # ── Intro ───────────────────────────────────────────────────────────
    cells.append(m(r"""
# How Chat Models Are Really Trained: Pretraining, SFT and DPO

The first GuppyLM notebook trained a model in one go. Real chat models go through **stages**. Each stage uses the same model and the same loop. Only the **data** and the **loss** change.

| Stage | Data | What the model learns | Status in this notebook |
|---|---|---|---|
| 0. Pretraining | Plain text, no chat format | The language itself: which words follow which | Done below |
| 1. SFT (supervised fine-tuning) | Chat conversations | To answer when asked (loss only on the answer) | Coming next |
| 2. DPO (direct preference optimization) | Pairs of a good and a bad answer | To prefer the good one | Coming after SFT |

You will chat with the model after every stage, so you can see what each one adds.

**Honest limits.** The data is synthetic fish talk made from templates, so every stage here is a miniature of the real thing. Real pretraining uses trillions of tokens of web text. The mechanics are the same, the scale is not.

**How to run it.** Pick a GPU (**Runtime > Change runtime type**), then run the cells from top to bottom.
"""))

    # ── 1. Setup ────────────────────────────────────────────────────────
    cells.append(m(r"""
## 1. Setup
"""))
    cells.append(c(r"""
!pip install -q tokenizers datasets

import os, shutil, time, json, random
import torch

# Start from an empty folder so old files from earlier runs cannot confuse us.
if os.path.exists('/content/guppy'):
    shutil.rmtree('/content/guppy')
os.makedirs('/content/guppy')
os.chdir('/content/guppy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch {torch.__version__}, device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU found. Training on CPU is about 20 times slower.')
"""))

    # ── 2. Source files ─────────────────────────────────────────────────
    cells.append(m(r"""
## 2. The source files

These are the real GuppyLM files. `config.py`, `model.py`, `dataset.py` and `train.py` are the same ones the first notebook used (`dataset.py` and `train.py` are only needed here because `stages.py` borrows `get_lr` from `train.py`).
The new file is **`stages.py`**. It holds everything the stages share: a packed-text dataset, loss functions, and one training loop that every stage reuses. Read its top comment first.
"""))
    for name in ["config.py", "model.py", "dataset.py", "train.py", "stages.py"]:
        cells.append(code(f"%%writefile {name}\n" + src(name)))

    # ── 3. Data ─────────────────────────────────────────────────────────
    cells.append(m(r"""
## 3. Data for stage 0: plain sentences

We reuse the 57,000 fish conversations, but we **take them apart**. Every user sentence and every Guppy sentence becomes a separate piece of plain text. There is no chat format and no pairing of a question with its answer.

This is the point of pretraining: the model reads lots of text and learns how the language works. It does not learn yet that a question should get an answer. That comes from SFT.

We also do three things real pipelines do:
1. **Deduplicate.** Exact repeats waste training time. They would also leak between the training and evaluation sets.
2. **Shuffle**, so neighbouring sentences have nothing to do with each other.
3. **Hold out 2%** for evaluation.
"""))
    cells.append(c(r"""
from datasets import load_dataset
from stages import write_pretrain_corpus

ds = load_dataset('arman-bd/guppylm-60k-generic')
rows = ds['train']

# One list with every user sentence and every Guppy sentence, in no particular order.
sentences = [r['input'] for r in rows] + [r['output'] for r in rows]
print(f'{len(sentences):,} sentences before removing duplicates')

n_train, n_eval = write_pretrain_corpus(sentences, 'data/pretrain_train.jsonl', 'data/pretrain_eval.jsonl')
print(f'{n_train + n_eval:,} unique sentences: {n_train:,} for training, {n_eval:,} held out for evaluation')

print('\nFirst five lines of the corpus (note: unrelated to each other):')
with open('data/pretrain_train.jsonl') as f:
    for _, line in zip(range(5), f):
        print('  ', json.loads(line)['text'])
"""))

    # ── 4. Tokenizer ────────────────────────────────────────────────────
    cells.append(m(r"""
## 4. The tokenizer (one tokenizer for all stages)

All stages must use the **same** tokenizer, because the model's weights only make sense for the token ids it was trained with. So we train the tokenizer once, now, on text in chat format (the format stages 1 and 2 will use), and reuse it everywhere. Chat markers like `<|im_start|>` become single tokens.
"""))
    cells.append(c(r"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# The chat-format text is used ONLY to train the tokenizer here.
chat_texts = [
    f'<|im_start|>user\n{r["input"]}<|im_end|>\n<|im_start|>assistant\n{r["output"]}<|im_end|>'
    for r in rows
]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=['<pad>', '<|im_start|>', '<|im_end|>'],   # ids 0, 1, 2
    min_frequency=2,
    show_progress=False,
)
tokenizer.train_from_iterator(chat_texts, trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.save('data/tokenizer.json')
print(f'Vocabulary size: {tokenizer.get_vocab_size()}')
"""))

    # ── 5. Packing ──────────────────────────────────────────────────────
    cells.append(m(r"""
## 5. Packing text into blocks

The first notebook padded short sentences up to the longest one in each batch. Pretraining does it differently: it **glues all texts into one long stream** and cuts the stream into blocks of exactly 128 tokens. Nothing is wasted on padding.

Texts are separated by the end-of-text token (id 2, the same id as `<|im_end|>`). Here it simply means "this text is over", and the model learns to write it when a sentence is finished.

Let's look at one block. The `y` row is the `x` row shifted by one token, which is the whole training task: **predict the next token at every position**.
"""))
    cells.append(c(r"""
from stages import PackedDataset, get_pretrain_loader

BLOCK = 128
train_ds = PackedDataset('data/pretrain_train.jsonl', 'data/tokenizer.json', block_size=BLOCK)
print(f'{len(train_ds.tokens):,} tokens in the stream -> {len(train_ds):,} blocks of {BLOCK} tokens')

x, y = train_ds[0]
print('x[:12]:', x[:12].tolist())
print('y[:12]:', y[:12].tolist(), ' <- same numbers, one step to the left')

print('\nBlock 0 decoded (id 2 marks where one text ends and the next begins):')
text = tokenizer.decode(x[:60].tolist(), skip_special_tokens=False)
print(text.replace('<|im_end|>', ' ⏎ '))
"""))

    # ── 6. Train ────────────────────────────────────────────────────────
    cells.append(m(r"""
## 6. Pretrain

We use the same GuppyLM as before, built from scratch (random weights). Before training, the loss should be about `ln(4096) = 8.32`, the loss of a random guess.

`train_stage` is the loop from the first notebook with one change: the loss is passed in as a function (`lm_loss`). Stages 1 and 2 will pass in different loss functions and reuse the rest.

We train for 2,000 steps. The corpus is small (about 218,000 tokens, 53 batches per epoch), so 2,000 steps means the model reads it roughly 38 times. That is plenty, and in fact more than it needs, as the next cell shows.
"""))
    cells.append(c(r"""
from config import GuppyConfig
from model import GuppyLM
from stages import lm_loss, train_stage

config = GuppyConfig()
torch.manual_seed(42)
model = GuppyLM(config).to(device)
print(model.param_summary())

train_loader = get_pretrain_loader('data/pretrain_train.jsonl', 'data/tokenizer.json', BLOCK, batch_size=32, shuffle=True)
eval_loader = get_pretrain_loader('data/pretrain_eval.jsonl', 'data/tokenizer.json', BLOCK, batch_size=32, shuffle=False)
print(f'{len(train_loader)} training batches per epoch, {len(eval_loader)} evaluation batches')

model.eval()
with torch.no_grad():
    batch = [t.to(device) for t in next(iter(train_loader))]
    print(f'Loss before training: {lm_loss(model, batch).item():.3f}   (ln(4096) = 8.318)')
"""))
    cells.append(c(r"""
t0 = time.time()
losses, best_eval = train_stage(
    model, train_loader, eval_loader, lm_loss, device,
    save_path='checkpoints/pretrain/best.pt',
    max_steps=2000, learning_rate=3e-4, warmup_steps=100,
)
print(f'Total wall time: {time.time() - t0:.0f} s')
"""))
    cells.append(c(r"""
import matplotlib.pyplot as plt

def smooth(values, k=50):
    return [sum(values[max(0, i - k + 1): i + 1]) / len(values[max(0, i - k + 1): i + 1]) for i in range(len(values))]

plt.figure(figsize=(7, 3.5))
plt.plot(losses, alpha=0.25, label='per step')
plt.plot(smooth(losses), label='average of last 50 steps')
plt.xlabel('step'); plt.ylabel('training loss'); plt.legend(); plt.title('Stage 0: pretraining loss')
plt.show()
print(f'First loss: {losses[0]:.3f}   Last loss: {losses[-1]:.3f}   Best eval loss: {best_eval:.3f}')
"""))
    cells.append(m(r"""
**Read the table above, not only the plot.** The training loss keeps falling to the end. The eval loss, measured on sentences the model never saw, tells a different story: it reaches its lowest value around step 800, then climbs back up. That is **overfitting**: the model starts to memorise its training sentences instead of learning the language.

This is why `train_stage` saves the checkpoint with the best eval loss and not the last one. The file `checkpoints/pretrain/best.pt` holds the model from around step 800.

Two caveats. The evaluation set is small (a few hundred sentences), so the eval numbers are noisy. And the corpus is tiny and very repetitive: only about a tenth of the 114,000 sentences were unique. Real pretraining data is far larger and far more varied, so it takes much longer to overfit.
"""))

    # ── 7. What did it learn ────────────────────────────────────────────
    cells.append(m(r"""
## 7. What did pretraining teach?

Let's load the best checkpoint and see what the model does. A pretrained model is a **text continuer**: you give it the start of a text and it writes what probably comes next. It has never seen a question followed by an answer, so we expect it to continue sentences, not to answer them.
"""))
    cells.append(c(r"""
ckpt = torch.load('checkpoints/pretrain/best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded the checkpoint from step {ckpt['step']} (eval loss {ckpt['eval_loss']:.3f})")

def complete(text, max_new_tokens=30, temperature=0.8):
    # Encode the text, let the model add tokens, decode everything back to text.
    ids = tokenizer.encode(text).ids
    x = torch.tensor([ids], device=device)
    out, _ = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
    return tokenizer.decode(out[0].tolist())
"""))
    cells.append(c(r"""
print('--- Continuing a sentence ---')
for start in ['the water is', 'i love', 'my favorite thing', 'it is']:
    print(f'{start!r:>22} -> {complete(start)}')

print('\n--- Asking a question in chat format ---')
question = '<|im_start|>user\nare you hungry<|im_end|>\n<|im_start|>assistant\n'
for _ in range(3):
    print(complete(question, max_new_tokens=25).replace('\n', ' | '))
"""))
    cells.append(m(r"""
**What to look for.** The continuations should read like fish talk: short, lowercase, about water and food. That is what pretraining gives: the voice and the vocabulary.
The chat-format question is a different story. The model has not been taught to answer. In stage 1 (SFT) it will be.

**Where we are.** The pretrained weights are saved in `checkpoints/pretrain/best.pt`. Stage 1 starts from this file instead of from random weights. That is the key idea: **each stage starts where the previous one stopped**.
"""))

    # ── Tests ───────────────────────────────────────────────────────────
    cells.append(m(r"""
## Optional: run the unit tests

The repository has tests for `stages.py`: duplicate removal, block packing, and a short training run on a toy corpus. This cell downloads the repository and runs them.
"""))
    cells.append(c(r"""
!rm -rf /content/repo && git clone -q -b feature/beginner-notebook https://github.com/i3wangyi/guppylm /content/repo
!cd /content/repo && python -m unittest discover -s tests -v 2>&1 | grep -E "^(test_|Ran|OK|FAILED)"
"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "GuppyLM: Post-Training (Pretrain, SFT, DPO)", "gpuType": "L4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    write_notebook(build(), "guppylm_post_training.ipynb")
