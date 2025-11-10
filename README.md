# DeepSeek Minimal Implementation

This repository provides a compact, from-scratch implementation of a
DeepSeek-style mixture-of-experts transformer model using PyTorch. It includes a
simple character-level tokenizer and an example training script showing how to
train the model on a corpus of text files.

## Features

- Rotary position embeddings with RMSNorm pre-normalization.
- Multi-head self-attention.
- Mixture-of-Experts feed-forward block with token-level top-k routing and a
  shared expert path.
- Configurable hyperparameters via `DeepSeekConfig`.
- Character-level tokenizer without external dependencies.
- Example training script for quick experimentation.

## Quick Start

1. Install dependencies (PyTorch is required).

   ```bash
   pip install -e .
   ```

2. Prepare a directory with `.txt` files containing your training corpus.

3. Train the model:

   ```bash
   python examples/train_deepseek.py /path/to/text/files --epochs 3
   ```

4. Generate text using the trained model (example):

   ```python
   import torch
   from deepseek import DeepSeekCharTokenizer, DeepSeekConfig, DeepSeekModel

   tokenizer = DeepSeekCharTokenizer()
   config = DeepSeekConfig(vocab_size=tokenizer.vocab_size)
   model = DeepSeekModel(config)
   prompt = torch.tensor([tokenizer.encode("Hello", add_special_tokens=False)])
   generated = model.generate(prompt, max_new_tokens=10)
   print(tokenizer.decode(generated[0].tolist()))
   ```

> **Note:** This implementation is intended for educational purposes and small
> scale experiments. Training a production-grade DeepSeek model requires
> significant compute resources and additional optimizations not covered here.
