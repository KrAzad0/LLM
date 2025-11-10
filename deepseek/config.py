"""Configuration dataclasses for the DeepSeek model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepSeekConfig:
    """Configuration for the DeepSeek transformer model."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None
    intermediate_size: int = 4096
    activation: str = "silu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = False
    num_experts: int = 16
    experts_per_token: int = 2
    shared_expert_intermediate_size: Optional[int] = None
    shared_expert_factor: float = 0.5
    no_moe_layers: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.rotary_dim is None:
            self.rotary_dim = self.hidden_size // self.num_attention_heads
        if self.shared_expert_intermediate_size is None:
            self.shared_expert_intermediate_size = int(
                self.intermediate_size * self.shared_expert_factor
            )
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token cannot exceed num_experts")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_scaling(self) -> float:
        return (self.rotary_dim or 1) / self.head_dim
