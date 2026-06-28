# LLM from Scratch

A hands-on implementation of a GPT-style decoder-only language model built entirely in PyTorch — no high-level model wrappers. This project walks through every layer of the transformer stack, from tokenization to text generation, in a single Jupyter notebook.

**Repository:** [github.com/debanshu005/llm_from_scratch](https://github.com/debanshu005/llm_from_scratch)

---

## Overview

Large language models often feel like black boxes. This project demystifies them by building one piece by piece:

- Load and tokenize real text data
- Implement multi-head causal self-attention
- Stack transformer blocks with pre-norm residuals
- Train with AdamW, gradient clipping, and a warmup + cosine LR schedule
- Generate text with temperature and top-k sampling
- Save and reload checkpoints

The default setup trains a small GPT on [WikiText-2](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) using the GPT-2 tokenizer (`tiktoken`). It is sized to run on a laptop CPU, with clear knobs to scale up when you have more compute.

---

## Architecture

```
Token IDs
    │
    ▼
Token Embedding + Sinusoidal Positional Encoding
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Block (× N)            │
│  ┌───────────────────────────────┐  │
│  │ LayerNorm → Causal Attention  │  │
│  │           → Residual          │  │
│  │ LayerNorm → Feed-Forward      │  │
│  │           → Residual          │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
Final LayerNorm → LM Head (weight-tied with embeddings)
    │
    ▼
Next-token logits
```

### Components

| Component | Description |
|-----------|-------------|
| **CausalSelfAttention** | Multi-head masked self-attention with scaled dot-product scores |
| **FeedForward** | Two-layer MLP with GELU activation (4× expansion) |
| **TransformerBlock** | Pre-norm block: attention + MLP with residual connections |
| **GPT** | Full decoder-only model with weight tying and cross-entropy loss |
| **TokenBlockDataset** | Sliding-window dataset over pre-tokenized text |
| **WarmupCosineScheduler** | Linear warmup followed by cosine decay |
| **generate()** | Autoregressive sampling with temperature and top-k filtering |

### Default model config

| Parameter | Value |
|-----------|-------|
| Vocabulary | 50,257 (GPT-2 / tiktoken) |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dim | 128 |
| Context length | 128 tokens |
| Training examples | 2,000 train / 200 val |
| Optimizer | AdamW (lr=3e-4, weight decay=0.1) |
| Epochs | 5 |

---

## Project structure

```
llm_from_scratch/
├── llm_training.ipynb   # Full pipeline: data → model → train → generate
├── requirements.txt     # Pinned dependencies
├── checkpoints/         # Saved model weights (created after training)
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.10+
- ~2 GB disk space (PyTorch + datasets)

### Installation

```bash
git clone https://github.com/debanshu005/llm_from_scratch.git
cd llm_from_scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Optional — for progress bars in Jupyter:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Run the notebook

```bash
jupyter notebook llm_training.ipynb
```

Run all cells top to bottom. On CPU, training the default config takes several minutes.

### Device selection

The notebook auto-detects CUDA. For Apple Silicon, you can use MPS:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

---

## Training and generation

After training completes, the notebook will:

1. Plot train vs. validation loss
2. Generate text from a prompt (default: `"The history of artificial intelligence"`)
3. Save a checkpoint to `checkpoints/gpt_scratch.pt`

Example generation call:

```python
prompt_tokens = tokenizer.encode("The history of artificial intelligence")
context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

generated = generate(model, context, max_new_tokens=80, temperature=0.8, top_k=40)
print(tokenizer.decode(generated[0].tolist()))
```

---

## Scaling up

Once the pipeline runs end-to-end, increase quality by tuning these values in the notebook:

```python
num_train_examples = 10000
train_config = GPTConfig(
    vocab_size=tokenizer.n_vocab,
    n_layer=6,
    n_head=8,
    n_embd=256,
    seq_len=256,
)
epochs = 10
```

| Change | Effect |
|--------|--------|
| More training examples | Better generalization |
| Larger `n_embd` / `n_layer` | More model capacity |
| Longer `seq_len` | Wider context window |
| More epochs | Lower loss (watch for overfitting) |

---

## Tech stack

- [PyTorch](https://pytorch.org/) — model and training
- [tiktoken](https://github.com/openai/tiktoken) — GPT-2 tokenization
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — WikiText-2 loading
- [Matplotlib](https://matplotlib.org/) — loss curves

---

## What you'll learn

- How causal (masked) attention prevents the model from "cheating" on future tokens
- Why pre-norm and residual connections stabilize deep transformers
- How weight tying reduces parameters and often improves performance
- The full training loop: forward pass, loss, backward pass, gradient clipping, LR scheduling
- Practical text generation with temperature and top-k sampling

---

## License

This project is open source. Feel free to use, modify, and share it for learning purposes.

---

## Author

**Debanshu Biswas** — [GitHub](https://github.com/debanshu005)

If this project helped you understand how LLMs work under the hood, consider starring the repo.
