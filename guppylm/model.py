"""
GuppyLM — a tiny fish brain.

Vanilla transformer: multi-head attention, ReLU FFN, LayerNorm, learned positional embeddings.
No GQA, no SwiGLU, no parallel residual, no RoPE. As simple as it gets.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GuppyConfig


# Shape letters used in the comments below:
#   B = batch size (sentences at once), T = tokens per sentence, C = model width (d_model),
#   H = number of attention heads, D = size of one head (C / H).
class Attention(nn.Module):
    # Attention lets each token look at earlier tokens and mix in information from the ones that matter.
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads  # each head works on a slice of the vector

        # One Linear layer makes query, key and value together (3 x C outputs). Think of it as a lookup:
        #   query = "what am I looking for?", key = "what do I contain?", value = "what do I hand over?"
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)  # mixes the heads' results back together
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # (B, T, 3C) -> (B, T, 3, H, D) -> (3, B, H, T, D), so q, k and v each come out as (B, H, T, D).
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Score for every pair of tokens: how well does token i's query match token j's key?
        # Result shape (B, H, T, T). Dividing by sqrt(D) keeps the numbers small so softmax does not saturate.
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # Where the mask is 0 (a future token), set the score to -infinity.
            # Softmax turns -infinity into probability 0, so a token cannot peek at what comes after it.
            attn = attn.masked_fill(mask == 0, float("-inf"))
        # Softmax turns each row of scores into probabilities that add up to 1 (the "attention weights").
        attn = self.dropout(F.softmax(attn, dim=-1))
        # Weighted average of the values, then put the heads side by side again: (B, H, T, D) -> (B, T, C).
        return self.out((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class FFN(nn.Module):
    # Feed-forward network: processes each token on its own. Attention mixes information BETWEEN tokens,
    # this part thinks about each token by itself. It widens the vector (C -> 768), applies ReLU, narrows it back.
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)
        self.down = nn.Linear(config.ffn_hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # ReLU(x) = max(0, x). Without a non-linear step like this, stacked Linear layers would collapse into one.
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    # One transformer block: attention, then FFN. The model stacks n_layers of these.
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, mask=None):
        # "x = x + ..." is a residual connection: the block adds a correction to its input instead of
        # replacing it. That keeps the original signal intact and makes deep stacks trainable.
        # LayerNorm rescales each token's vector to a steady size before each sub-layer.
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GuppyLM(nn.Module):
    def __init__(self, config: GuppyConfig):
        super().__init__()
        self.config = config

        # An Embedding is a table: token id -> vector of C numbers. The numbers are learned in training.
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Attention itself has no sense of order, so we also add a learned vector for each POSITION (0, 1, 2, ...).
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        # The last layer turns a vector of C numbers into one score per vocabulary token ("logits").
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: the output layer reuses the embedding table. It saves about 1.6M parameters and
        # makes sense, since "id -> vector" and "vector -> id" are mirror images of each other.
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        self.apply(self._init_weights)  # call _init_weights on every sub-module

    def _init_weights(self, m):
        # Start from small random numbers (standard deviation 0.02) and zero biases.
        # Big random starting weights make early training unstable.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: token ids, shape (B, T). targets: the "next token" answers, same shape (only used in training).
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)                  # 0, 1, 2, ..., T-1
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))      # (B, T, C): what the token is + where it is
        # Lower-triangular matrix of ones: row i has 1s only for columns 0..i (the tokens up to and including i).
        # The two unsqueeze calls add the batch and head dimensions so the mask fits (B, H, T, T) scores.
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(self.norm(x))                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy compares the predicted scores with the true next token. A perfect guess gives 0;
            # a random guess over 4096 tokens gives about ln(4096) = 8.3.
            # view(-1, ...) flattens (B, T) into one long list of predictions.
            # ignore_index=0 skips <pad> positions, so filler tokens are not counted.
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    # @torch.no_grad() turns off gradient tracking. We only generate here, not train, so it saves memory and time.
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50, **kwargs):
        self.eval()  # switch off dropout
        for _ in range(max_new_tokens):
            # The model can only read max_seq_len tokens, so keep just the most recent ones.
            idx_cond = idx[:, -self.config.max_seq_len:]
            # Run the whole sequence again for every new token. (Real systems cache earlier work, the "KV cache".
            # We skip that to keep the code simple.)
            logits, _ = self(idx_cond)
            # Only the LAST position matters: it holds the scores for the next token.
            # Dividing by temperature: below 1 makes the top choice even more likely, above 1 flattens the odds.
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                # Top-k sampling: keep only the k best-scoring tokens and rule out the rest, so a very
                # unlikely token cannot be picked by accident.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)                       # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)       # draw one token at random using those odds
            idx = torch.cat([idx, next_id], dim=1)                  # append it and go around again
            # .item() only works when there is exactly one sentence (batch size 1), which is how we chat.
            if next_id.item() == self.config.eos_id:
                break                                               # the model wrote <|im_end|>: it is done
        return idx, []

    def param_count(self):
        # parameters() lists every learnable tensor. Tied weights are the same tensor, so they count once.
        total = sum(p.numel() for p in self.parameters())
        return total, 0

    def param_summary(self):
        total, _ = self.param_count()
        return f"GuppyLM: {total:,} params ({total/1e6:.1f}M)"
