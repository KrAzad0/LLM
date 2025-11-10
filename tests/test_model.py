"""Basic unit tests for the DeepSeek model."""

from __future__ import annotations

import torch

from deepseek import DeepSeekCharTokenizer, DeepSeekConfig, DeepSeekModel


def test_forward_shape() -> None:
    tokenizer = DeepSeekCharTokenizer()
    config = DeepSeekConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        num_experts=4,
        experts_per_token=2,
        max_position_embeddings=64,
    )
    model = DeepSeekModel(config)
    inputs = torch.randint(0, tokenizer.vocab_size, (2, 16))
    logits = model(inputs)
    assert logits.shape == (2, 16, tokenizer.vocab_size)


def test_generate_appends_tokens() -> None:
    tokenizer = DeepSeekCharTokenizer()
    config = DeepSeekConfig(vocab_size=tokenizer.vocab_size)
    model = DeepSeekModel(config)
    prompt = torch.randint(0, tokenizer.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=4)
    assert generated.shape[-1] == prompt.shape[-1] + 4
