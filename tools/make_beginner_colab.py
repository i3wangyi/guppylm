"""Generate the beginner-friendly GuppyLM Colab notebook.

Run:  python3 tools/make_beginner_colab.py
Output: guppylm_beginner.ipynb (repo root)

The notebook writes the real source files from guppylm/ into Colab, so it never
drifts from the package. Only the explanations and small experiments live here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_colab import PROJECT_ROOT, code, md, read_for_colab, write_notebook  # noqa: E402


def src(name):
    return read_for_colab(os.path.join(PROJECT_ROOT, "guppylm", name))


def c(text):
    return code(text.strip("\n"))


def m(text):
    return md(text.strip("\n"))


def build():
    cells = []

    # ── 0. Intro ────────────────────────────────────────────────────────
    cells.append(m(r"""
# Build a Language Model From Scratch: GuppyLM

In this notebook you will train a small language model (LLM) that talks like a fish.
You will see every step: the data, the tokenizer, the model, the training loop, and text generation.

GuppyLM has about 9 million parameters. GPT-style models with billions of parameters work the same way, only bigger.
So if you understand this notebook, you understand the core of how a chat model is built.

**How to run it**
1. Open the menu **Runtime > Change runtime type** and pick a GPU (see the note below).
2. Run the cells from top to bottom with **Shift+Enter**. Read the text above each cell first.
3. Total time: about 6 minutes on an L4 GPU (training itself takes about 4.5 minutes).

**Which GPU?** This model is tiny: 8.7 million parameters and batches of 32 short sentences. It needs very little GPU memory, and a bigger GPU helps less than you might expect, because a lot of each training step goes to Python overhead and launching many small GPU jobs.
An **L4** is a good pick. It is fast enough (10,000 training steps in about 4.5 minutes) and it uses far fewer Colab compute units per hour than an A100 or H100.
A free **T4** also works.

**Map of this notebook**

| Step | What you do | Idea you learn |
|---|---|---|
| 1 | Check the GPU | Why GPUs matter |
| 2 | Write the source files | Where the code lives |
| 3 | Look at the data | A model learns from examples |
| 4 | Train a tokenizer | Text becomes numbers |
| 5 | Build the model | Embeddings, attention, tensor shapes |
| 6 | Train on one batch | How learning works (loss, gradients) |
| 7 | Train for real | Learning rate schedule, loss curve |
| 8 | Talk to Guppy | Sampling and temperature |
| 9 | Experiments | Change something and watch what happens |
"""))

    # ── 1. Setup ────────────────────────────────────────────────────────
    cells.append(m(r"""
## 1. Setup and GPU check

We install the libraries and make a clean working folder. Then we check which device PyTorch can use.

A language model is mostly big matrix multiplications. A GPU runs thousands of small multiplications at the same time, so it is much faster than a CPU for this work.
"""))
    cells.append(c(r"""
!pip install -q tokenizers datasets

import os, shutil, time
import torch

# Start from an empty folder so old files from earlier runs cannot confuse us.
if os.path.exists('/content/guppy'):
    shutil.rmtree('/content/guppy')
os.makedirs('/content/guppy')
os.chdir('/content/guppy')

print(f'PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU found. Use Runtime > Change runtime type, then pick a GPU. Training on CPU takes much longer.')
"""))
    cells.append(m(r"""
Let's measure the difference. The cell below multiplies two 2048 x 2048 matrices ten times, on the CPU and then on the GPU.
"""))
    cells.append(c(r"""
def time_matmul(device, n=2048, repeats=10):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b                                   # warm-up run, not timed
    if device == 'cuda':
        torch.cuda.synchronize()            # GPU work is asynchronous: wait for it to finish
    t0 = time.time()
    for _ in range(repeats):
        a @ b
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats

cpu_t = time_matmul('cpu')
print(f'CPU: {cpu_t*1000:.1f} ms per multiplication')
if torch.cuda.is_available():
    gpu_t = time_matmul('cuda')
    print(f'GPU: {gpu_t*1000:.1f} ms per multiplication  ({cpu_t/gpu_t:.0f}x faster)')
"""))

    # ── 2. Source files ─────────────────────────────────────────────────
    cells.append(m(r"""
## 2. The source files

The next five cells write the real GuppyLM code to disk. Each file has one job.
You do not need to understand every line now. Each section below points back to the part that matters.

| File | Job |
|---|---|
| `config.py` | All the knobs: model size, learning rate, number of steps |
| `model.py` | The transformer itself |
| `dataset.py` | Turns text into batches of numbers |
| `train.py` | The training loop |
| `inference.py` | Loads a trained model and generates replies |

`%%writefile` is a Colab trick: it saves the cell's content as a file instead of running it.
"""))
    cells.append(m(r"""
**`config.py`**. Two small classes that only hold numbers. Look at `GuppyConfig`: `d_model=384` is the width of every vector inside the model, `n_layers=6` is how many transformer blocks are stacked, and `vocab_size=4096` is how many different tokens exist.
"""))
    cells.append(code("%%writefile config.py\n" + src("config.py").strip("\n")))
    cells.append(m(r"""
**`model.py`**. The heart of the project, about 130 lines. We take it apart in step 5, so just skim it now.
"""))
    cells.append(code("%%writefile model.py\n" + src("model.py").strip("\n")))
    cells.append(m(r"""
**`dataset.py`**. Notice `x = ids[:-1]` and `y = ids[1:]`. The target `y` is the input `x` shifted by one token. That one line is the whole training task: **predict the next token**.
"""))
    cells.append(code("%%writefile dataset.py\n" + src("dataset.py").strip("\n")))
    cells.append(m(r"""
**`train.py`**. The loop that repeats: get a batch, compute the loss, compute gradients, update the weights. We run it in step 7.
"""))
    cells.append(code("%%writefile train.py\n" + src("train.py").strip("\n")))
    cells.append(m(r"""
**`inference.py`**. Loads a saved model and turns a user message into a reply.
"""))
    cells.append(code("%%writefile inference.py\n" + src("inference.py").strip("\n")))

    # ── 3. Data ─────────────────────────────────────────────────────────
    cells.append(m(r"""
## 3. The data

A language model learns from examples. GuppyLM uses 60,000 short conversations between a user and a fish, written by a script and shared on HuggingFace.

We save each conversation in **ChatML** format, a plain-text layout many chat models use:

```
<|im_start|>user
hi guppy<|im_end|>
<|im_start|>assistant
hello. the water is nice today.<|im_end|>
```

`<|im_start|>` and `<|im_end|>` are special markers. They tell the model where a message starts and ends. The model learns to write `<|im_end|>` when its answer is finished.
"""))
    cells.append(c(r"""
import json, random
from collections import Counter
from datasets import load_dataset

ds = load_dataset('arman-bd/guppylm-60k-generic')
print(ds)

os.makedirs('data', exist_ok=True)
texts = []
for split, path in [('train', 'data/train.jsonl'), ('test', 'data/eval.jsonl')]:
    with open(path, 'w') as f:
        for row in ds[split]:
            text = (
                f'<|im_start|>user\n{row["input"]}<|im_end|>\n'
                f'<|im_start|>assistant\n{row["output"]}<|im_end|>'
            )
            f.write(json.dumps({'text': text, 'category': row['category']}) + '\n')
            texts.append(text)
    print(f'{path}: {len(ds[split]):,} conversations')

print('\nFive random examples:\n')
random.seed(0)
for row in random.sample(list(ds['train']), 5):
    print(f'[{row["category"]}]')
    print(f'  user:   {row["input"]}')
    print(f'  guppy:  {row["output"]}\n')

cats = Counter(ds['train']['category'])
print(f'{len(cats)} topics, for example: {cats.most_common(5)}')
"""))

    # ── 4. Tokenizer ────────────────────────────────────────────────────
    cells.append(m(r"""
## 4. The tokenizer: text to numbers

A neural network only understands numbers. A **tokenizer** cuts text into pieces called tokens and gives each piece an integer id.

We use **BPE** (byte-pair encoding). It starts from single characters and keeps merging the pairs that appear most often in the training text. Common words end up as one token, rare words get split into several pieces.

We ask for a vocabulary of up to 4,096 tokens. Three of them are reserved for special markers: `<pad>` (id 0, filler to make batches the same length), `<|im_start|>` (id 1) and `<|im_end|>` (id 2).
"""))
    cells.append(c(r"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=['<pad>', '<|im_start|>', '<|im_end|>'],
    min_frequency=2,
    show_progress=False,
)
tokenizer.train_from_iterator(texts, trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.save('data/tokenizer.json')
print(f'Vocabulary size: {tokenizer.get_vocab_size()}')
"""))
    cells.append(m(r"""
**Why is the vocabulary smaller than 4,096?** The fish only talks about a few hundred words. The trainer stops merging when no pair of pieces appears at least twice, so it ends with fewer tokens than the limit. The model still allocates 4,096 slots, and the unused ones just stay untrained. That is why the loss check in step 5 uses 4,096.

Now play with it. Change the sentence and run the cell again. Words the fish knows well (water, food) become one token. Words it has never seen, like "quantum", split into smaller pieces.
The `Ġ` you see at the start of a token is how this tokenizer writes a leading space, so `Ġlove` means " love".
"""))
    cells.append(c(r"""
sentence = 'i love food and the water is warm'   # <- try your own sentence

enc = tokenizer.encode(sentence)
print('tokens:', enc.tokens)
print('ids:   ', enc.ids)
print('decoded back:', tokenizer.decode(enc.ids))

print('\nA word the fish never saw:')
enc2 = tokenizer.encode('quantum')
print(enc2.tokens)
"""))

    # ── 5. Model ────────────────────────────────────────────────────────
    cells.append(m(r"""
## 5. The model

Now we build the model and follow one batch of data through it, looking at the **shape** of the tensor at each stage.
A tensor is a multi-dimensional array (like a NumPy array that can also live on the GPU). Shapes are the easiest way to understand what a network does.

We use these letters:
- `B`: batch size, how many sentences we process at once
- `T`: sequence length, how many tokens in each sentence
- `C`: the model width (`d_model`, 384 here)
"""))
    cells.append(c(r"""
import torch.nn.functional as F
from config import GuppyConfig
from model import GuppyLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GuppyConfig()
model = GuppyLM(config).to(device)
print(model.param_summary())
print(f'{config.n_layers} layers, {config.n_heads} attention heads, width {config.d_model}')

# Take a real sentence, turn it into ids, and add a batch dimension.
prompt = '<|im_start|>user\nhi guppy<|im_end|>\n<|im_start|>assistant\n'
ids = tokenizer.encode(prompt).ids
x = torch.tensor([ids], device=device)
print(f'\nInput ids shape (B, T): {tuple(x.shape)}')

model.eval()
with torch.no_grad():
    B, T = x.shape
    tok = model.tok_emb(x)
    print(f'1. Token embedding  (B, T, C): {tuple(tok.shape)}   <- each id becomes a vector of {config.d_model} numbers')
    pos = model.pos_emb(torch.arange(T, device=device))
    print(f'2. Position embedding (T, C):  {tuple(pos.shape)}   <- tells the model where each token sits')
    h = tok + pos
    mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
    for i, block in enumerate(model.blocks):
        h = block(h, mask)
    print(f'3. After {config.n_layers} blocks (B, T, C): {tuple(h.shape)}   <- same shape, richer meaning')
    logits = model.lm_head(model.norm(h))
    print(f'4. Logits (B, T, vocab): {tuple(logits.shape)}   <- one score per possible next token, at every position')
"""))
    cells.append(m(r"""
**What happens inside a block?** Each of the 6 blocks does two things:

1. **Attention.** Every token looks at the earlier tokens and decides which ones matter for it. The model has 6 attention heads, so it can look for 6 different kinds of relationships at once.
2. **Feed-forward network (FFN).** A small two-layer network that processes each token on its own.

Both steps use a **residual connection** (`x = x + ...`): the block adds its result to the input instead of replacing it. This keeps information flowing and makes deep networks easier to train.

**The causal mask.** When the model predicts token 5, it must not peek at token 6. A triangular mask hides the future. Here is what it looks like for 6 tokens (1 means "can look at"):
"""))
    cells.append(c(r"""
print(torch.tril(torch.ones(6, 6)).int())
"""))
    cells.append(m(r"""
**A sanity check.** The model is untrained, so its guesses are random. With 4,096 possible tokens, a random guess gives the right token with probability 1/4096.
The loss (cross-entropy) for that is `ln(4096)`, about 8.32. If your first loss is near this number, the setup is correct.
"""))
    cells.append(c(r"""
import math
from dataset import get_dataloader

eval_loader = get_dataloader('data/eval.jsonl', 'data/tokenizer.json', config.max_seq_len, batch_size=32, shuffle=False)
x_batch, y_batch = next(iter(eval_loader))
x_batch, y_batch = x_batch.to(device), y_batch.to(device)
print('Batch shapes (B, T):', tuple(x_batch.shape), tuple(y_batch.shape))

model.eval()
with torch.no_grad():
    _, loss = model(x_batch, y_batch)
print(f'Loss of the untrained model: {loss.item():.3f}')
print(f'ln(4096) = {math.log(4096):.3f}')
"""))

    # ── 6. One batch ────────────────────────────────────────────────────
    cells.append(m(r"""
## 6. How learning works: overfit one batch

Training repeats four steps:

1. **Forward pass.** Run the batch through the model and get a loss, a single number that says how wrong the predictions are.
2. **Backward pass.** `loss.backward()` computes, for every weight, how much a small change would change the loss. These are the **gradients**.
3. **Update.** The optimizer (AdamW) nudges each weight a little in the direction that lowers the loss.
4. **Reset.** Clear the gradients and repeat.

Before the full run, let's watch this on a single batch. We train on the same 32 sentences again and again. The loss should fall fast, because the model can simply memorise them. (This is called **overfitting**. It is a useful debugging trick: if a model cannot even memorise one batch, something is broken.)
"""))
    cells.append(c(r"""
torch.manual_seed(0)
tiny_model = GuppyLM(config).to(device)
opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
tiny_model.train()

for step in range(61):
    _, loss = tiny_model(x_batch, y_batch)   # 1. forward
    loss.backward()                          # 2. backward
    opt.step()                               # 3. update
    opt.zero_grad()                          # 4. reset
    if step % 10 == 0:
        print(f'step {step:3d}  loss {loss.item():.3f}')

del tiny_model, opt
"""))

    # ── 7. Full training ────────────────────────────────────────────────
    cells.append(m(r"""
## 7. Train for real

The real loop in `train.py` does the same four steps for 10,000 steps, with batches of 32 conversations, and adds a few things:

- **Learning rate schedule.** The learning rate is how big each nudge is. It starts at 0, climbs for 200 steps (**warmup**), then slowly falls along a cosine curve. Big steps early, careful steps late.
- **Gradient clipping.** If a gradient is huge, it is scaled down so one bad batch cannot wreck the model.
- **Mixed precision.** On a GPU, many calculations use 16-bit numbers instead of 32-bit. That is faster and uses less memory.
- **Evaluation.** Every 200 steps the model is tested on data it has never trained on. The best checkpoint is saved. If the training loss keeps falling while the eval loss rises, the model is memorising instead of learning.

Here is the learning rate schedule:
"""))
    cells.append(c(r"""
import matplotlib.pyplot as plt
from config import TrainConfig
from train import get_lr

tc = TrainConfig()
steps = list(range(tc.max_steps))
plt.figure(figsize=(7, 3))
plt.plot(steps, [get_lr(s, tc) for s in steps])
plt.xlabel('step'); plt.ylabel('learning rate'); plt.title('Warmup, then cosine decay')
plt.show()
"""))
    cells.append(m(r"""
Now run the training. It takes about 4.5 minutes on an L4. It prints a row every 100 steps. Watch the `Train` column fall from about 8.3 to about 0.4, and watch the `Eval` column follow it.
"""))
    cells.append(c(r"""
del model          # free the GPU memory used by the untrained model from step 5
from train import train

t0 = time.time()
train()
print(f'Total wall time: {time.time() - t0:.0f} s')
"""))
    cells.append(m(r"""
Training saved three kinds of files in `checkpoints/`: `best_model.pt` (lowest eval loss), `step_*.pt` (snapshots) and `final_model.pt` (the last step, with the full loss history). Let's plot the history.
"""))
    cells.append(c(r"""
ckpt = torch.load('checkpoints/final_model.pt', map_location='cpu', weights_only=False)
losses = ckpt['train_losses']

def smooth(values, k=100):
    return [sum(values[max(0, i - k + 1): i + 1]) / len(values[max(0, i - k + 1): i + 1]) for i in range(len(values))]

plt.figure(figsize=(7, 3.5))
plt.plot(losses, alpha=0.25, label='per step')
plt.plot(smooth(losses), label='average of last 100 steps')
plt.xlabel('step'); plt.ylabel('training loss'); plt.legend(); plt.title('Training loss')
plt.show()
print(f'First loss: {losses[0]:.3f}   Last loss: {losses[-1]:.3f}')
"""))
    cells.append(m(r"""
The loss drops steeply in the first 1,000 steps, then flattens around 0.4. It does not reach 0 most likely because the same kind of question has many different valid replies in the data, so the model cannot predict the exact next word every time. It only needs to give good odds to each valid one.
"""))

    # ── 8. Generation ───────────────────────────────────────────────────
    cells.append(m(r"""
## 8. Talk to Guppy

To generate text, the model runs in a loop:

1. Feed in the tokens so far.
2. Take the scores for the **next** token and turn them into probabilities with softmax.
3. Pick one token, append it, and go back to step 1.
4. Stop when the model writes `<|im_end|>`.

Let's load the best checkpoint and chat.
"""))
    cells.append(c(r"""
from inference import GuppyInference

engine = GuppyInference('checkpoints/best_model.pt', 'data/tokenizer.json', device=device)

def chat(text, temperature=0.7):
    r = engine.chat_completion([{'role': 'user', 'content': text}], temperature=temperature, max_tokens=64)
    return r['choices'][0]['message']['content']

for q in ['hi guppy', 'are you hungry', 'do you like bubbles', 'what is the meaning of life', 'tell me a joke', 'goodnight guppy']:
    print(f'You>   {q}')
    print(f'Guppy> {chat(q)}\n')
"""))
    cells.append(m(r"""
**What does the model actually "think"?** At each step it gives a probability to every token in its vocabulary. Here are its five favourite choices for the first word of the reply:
"""))
    cells.append(c(r"""
prompt = '<|im_start|>user\nare you hungry<|im_end|>\n<|im_start|>assistant\n'
x = torch.tensor([tokenizer.encode(prompt).ids], device=device)

engine.model.eval()
with torch.no_grad():
    logits, _ = engine.model(x)
probs = F.softmax(logits[0, -1], dim=-1)
top = probs.topk(5)
for p, i in zip(top.values, top.indices):
    print(f'{p.item()*100:5.1f}%  {tokenizer.decode([i.item()])!r}')
"""))
    cells.append(m(r"""
**Temperature** controls how adventurous the sampling is. The model divides its scores by the temperature before softmax:

- Low (0.1): the likeliest token almost always wins. Safe and repetitive.
- Medium (0.7): a good mix. This is the default.
- High (1.5 and up): unlikely tokens get a real chance. More surprising, and eventually nonsense.

Run the cell a few times. The 0.1 answers barely change. This model is small and its data is narrow, so even 1.5 stays fairly readable. Try 3.0 to see it fall apart.
"""))
    cells.append(c(r"""
question = 'what do you think about the rain'   # <- try your own question

for temp in [0.1, 0.7, 1.5, 3.0]:
    print(f'--- temperature {temp} ---')
    for _ in range(3):
        print('  ', chat(question, temperature=temp))
"""))

    # ── 9. Experiments ──────────────────────────────────────────────────
    cells.append(m(r"""
## 9. Where to go next

You trained a language model from raw text. The only difference between this and a large model is scale: more data, more parameters, more GPUs. Here are experiments to try. For each one, edit the file, retrain, and compare the loss curve and the replies.

1. **Train less.** In `config.py`, set `max_steps` to 1000. How much worse is Guppy? Where does the eval loss stop improving?
2. **Make the model smaller.** Set `n_layers` to 2 and `d_model` to 128. Check the parameter count and the quality.
3. **Break it on purpose.** Set `dropout` to 0 and `weight_decay` to 0, and train for 10,000 steps. Does the eval loss start rising while the train loss keeps falling?
4. **Change the learning rate.** Try `learning_rate = 3e-2` (100x too big). What happens to the loss?
5. **Ask it something it never saw.** "What is bitcoin?" Guppy was trained on fish topics only, so see what it makes up.

To edit a file in Colab, open the folder icon on the left, go to `guppy/`, and double-click the file. Do not re-run the `%%writefile` cells afterwards, because they would overwrite your edit.

Training writes into `checkpoints/`. To keep your first model for comparison, rename the folder first, then train again in a fresh process so it picks up your edit:

```
!mv checkpoints checkpoints_run1
!python train.py
```

**Keep your model.** The cell below packs the files you need into one archive. Uncomment the last two lines to download it to your computer.
"""))
    cells.append(c(r"""
!cd /content && tar czf guppylm.tar.gz guppy/checkpoints/best_model.pt guppy/checkpoints/config.json guppy/data/tokenizer.json guppy/model.py guppy/config.py guppy/inference.py
print(f'Archive size: {os.path.getsize("/content/guppylm.tar.gz") / 1e6:.1f} MB')

# from google.colab import files
# files.download('/content/guppylm.tar.gz')
"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "GuppyLM: Beginner Guide", "gpuType": "L4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    write_notebook(build(), "guppylm_beginner.ipynb")
