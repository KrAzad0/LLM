"""Core PyTorch modules implementing the DeepSeek architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import DeepSeekConfig


@dataclass
class RotaryCache:
    cos: torch.Tensor
    sin: torch.Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class RotaryEmbedding(nn.Module):
    """Implements rotary position embeddings as used by DeepSeek."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> RotaryCache:
        if seq_len > self.max_seq_len_cached:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
            return RotaryCache(cos=cos, sin=sin)
        return RotaryCache(
            cos=self.cos_cached[:, :, :seq_len, :].to(device),
            sin=self.sin_cached[:, :, :seq_len, :].to(device),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cache: RotaryCache
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cache.cos, cache.sin
    q_1, q_2 = q[..., ::2], q[..., 1::2]
    k_1, k_2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack([q_1 * cos - q_2 * sin, q_1 * sin + q_2 * cos], dim=-1)
    k_rot = torch.stack([k_1 * cos - k_2 * sin, k_1 * sin + k_2 * cos], dim=-1)
    q_rot = q_rot.flatten(-2)
    k_rot = k_rot.flatten(-2)
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dropout = config.attention_dropout
        bias = config.use_bias

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.rotary_emb = RotaryEmbedding(
            dim=config.rotary_dim or self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.attn_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cache = self.rotary_emb(seq_len, hidden_states.device)
        q, k = apply_rotary_pos_emb(q, k, cache)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class TopKRouter(nn.Module):
    """Token-level top-k router used by the mixture-of-experts MLP."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden_states)
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        gate_scores = F.softmax(top_logits, dim=-1)
        return top_indices, gate_scores


class ExpertMLP(nn.Module):
    """Feed-forward network used as an expert."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        bias = config.use_bias
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        self.activation = getattr(F, config.activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SharedExpertMLP(nn.Module):
    """Shared expert that processes every token."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        bias = config.use_bias
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            bias=bias,
        )
        self.fc2 = nn.Linear(
            config.shared_expert_intermediate_size,
            config.hidden_size,
            bias=bias,
        )
        self.activation = getattr(F, config.activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MixtureOfExperts(nn.Module):
    """Mixture-of-Experts feed-forward layer."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(config.num_experts)])
        self.shared_expert = SharedExpertMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        top_indices, gate_scores = self.router(hidden_states)
        bsz, seq_len, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        flat_indices = top_indices.reshape(-1, self.router.top_k)
        flat_scores = gate_scores.reshape(-1, self.router.top_k)

        expert_outputs = torch.zeros_like(flat_states)
        for expert_id, expert in enumerate(self.experts):
            for slot in range(self.router.top_k):
                token_mask = flat_indices[:, slot] == expert_id
                if not token_mask.any():
                    continue
                expert_input = flat_states[token_mask]
                routed = expert(expert_input)
                scores = flat_scores[token_mask, slot].unsqueeze(-1)
                expert_outputs[token_mask] += routed * scores

        expert_outputs = expert_outputs.view(bsz, seq_len, hidden_size)
        shared_output = self.shared_expert(hidden_states)
        output = expert_outputs + shared_output
        return self.dropout(output)


class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig, layer_idx: int) -> None:
        super().__init__()
        self.use_moe = layer_idx not in config.no_moe_layers
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        if self.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = ExpertMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_normed = self.attn_norm(hidden_states)
        attn_output = self.attn(attn_normed, attention_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)

        ffn_normed = self.ffn_norm(hidden_states)
        mlp_output = self.mlp(ffn_normed)
        hidden_states = hidden_states + self.dropout(mlp_output)
        return hidden_states


class DeepSeekModel(nn.Module):
    """Full DeepSeek transformer language model."""

    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [DeepSeekBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = self._prepare_attn_mask(attention_mask)

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _prepare_attn_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        return (1.0 - attention_mask) * -1e9

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=-1)
        return generated
