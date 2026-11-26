"""GuppyLM configuration."""

from dataclasses import dataclass


# A dataclass is a Python class that mostly holds data. Python writes __init__ for you,
# so GuppyConfig(n_layers=2) works. (Java comparison: a record, or a plain class with fields.)
@dataclass
class GuppyConfig:
    # Shape of the model. Change these and the model gets bigger or smaller.
    vocab_size: int = 4096    # how many different tokens the model knows (tokenizer may use fewer)
    max_seq_len: int = 128    # longest text, in tokens, the model reads at once
    d_model: int = 384        # width: every token is represented by a vector of this many numbers
    n_layers: int = 6         # how many transformer blocks are stacked on top of each other
    n_heads: int = 6          # attention heads per block; d_model must divide evenly by this
    ffn_hidden: int = 768     # inner width of the feed-forward network (2 x d_model here)
    dropout: float = 0.1      # during training, randomly zero 10% of values to reduce memorising

    # Special tokens. These ids must match the order given to the tokenizer trainer.
    pad_id: int = 0           # <pad>: filler so every sentence in a batch has the same length
    bos_id: int = 1           # <|im_start|>: a message begins
    eos_id: int = 2           # <|im_end|>: a message ends (the model learns to emit this to stop)


@dataclass
class TrainConfig:
    # How training runs. None of these change the model's shape.
    batch_size: int = 32            # sentences processed together in one step
    learning_rate: float = 3e-4     # peak size of each weight update (3e-4 = 0.0003)
    min_lr: float = 3e-5            # the learning rate falls to this at the end of training
    weight_decay: float = 0.1       # gently pulls weights toward 0, which helps against memorising
    warmup_steps: int = 200         # steps spent raising the learning rate from 0 to its peak
    max_steps: int = 10000          # total number of weight updates
    eval_interval: int = 200        # test on held-out data every this many steps
    save_interval: int = 500        # save a snapshot every this many steps
    grad_clip: float = 1.0          # cap on the total gradient size, so one bad batch cannot wreck the model
    device: str = "auto"            # "auto" picks cuda, then mps (Apple), then cpu
    seed: int = 42                  # fixes the random starting point so runs can be repeated
    data_dir: str = "data"          # folder with train.jsonl, eval.jsonl and tokenizer.json
    output_dir: str = "checkpoints" # folder where trained weights are saved
