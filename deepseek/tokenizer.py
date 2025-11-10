"""Simple character-level tokenizer for experimentation."""

from __future__ import annotations

from typing import Iterable, List


class DeepSeekCharTokenizer:
    """A minimal character-level tokenizer.

    This tokenizer is intentionally simple so the project has zero external
    dependencies. It provides encode/decode helpers compatible with the model.
    """

    def __init__(self, extra_tokens: Iterable[str] | None = None) -> None:
        base_vocab = [chr(i) for i in range(32, 127)]
        self.special_tokens = ["<pad>", "<bos>", "<eos>"]
        if extra_tokens is None:
            extra_tokens = []
        self.vocab = self.special_tokens + base_vocab + list(extra_tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [self.token_to_id.get(char, self.token_to_id["<pad>"]) for char in text]
        if add_special_tokens:
            return [self.token_to_id["<bos>"]] + tokens + [self.token_to_id["<eos>"]]
        return tokens

    def decode(self, token_ids: Iterable[int]) -> str:
        chars: List[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "")
            if token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def pad(self, sequences: List[List[int]], pad_token: str = "<pad>") -> List[List[int]]:
        pad_id = self.token_to_id[pad_token]
        max_len = max(len(seq) for seq in sequences)
        return [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
