"""Example training script for the DeepSeek model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from deepseek import DeepSeekCharTokenizer, DeepSeekConfig, DeepSeekModel


def read_corpus(files: Iterable[Path]) -> str:
    texts: List[str] = []
    for path in files:
        texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts)


class TextDataset(Dataset[int]):
    def __init__(self, tokenized: List[int], seq_len: int) -> None:
        self.tokens = tokenized
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.tokens[idx : idx + self.seq_len + 1]
        return torch.tensor(window, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([item[:-1] for item in batch])
    labels = torch.stack([item[1:] for item in batch])
    return inputs, labels


def create_model(tokenizer: DeepSeekCharTokenizer) -> DeepSeekModel:
    config = DeepSeekConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        num_experts=4,
        experts_per_token=2,
        max_position_embeddings=512,
    )
    return DeepSeekModel(config)


def train(args: argparse.Namespace) -> None:
    tokenizer = DeepSeekCharTokenizer()
    text = read_corpus(Path(args.data_dir).glob("*.txt"))
    tokenized = tokenizer.encode(text, add_special_tokens=False)

    dataset = TextDataset(tokenized, seq_len=args.sequence_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "deepseek.pt")
        print(f"Model saved to {output_dir / 'deepseek.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeepSeek model from scratch")
    parser.add_argument("data_dir", type=str, help="Directory containing .txt files")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
