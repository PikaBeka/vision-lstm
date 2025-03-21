# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

import torch
from torch import nn

from ...components.init import bias_linspace_init_
from ...components.ln import MultiHeadLayerNorm
from .backends import parallel_scan_log, log_g
import einops
import torch.nn.functional as F


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1
    bias: bool = False


class mLSTMCell(nn.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config

        self.backend_fn = parallel_scan_log

        # self.igate = nn.Linear(3 * config.embedding_dim, config.num_heads)
        # self.fgate = nn.Linear(3 * config.embedding_dim, config.num_heads)

        self.linear_i = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=1, groups=config.embedding_dim, bias=False)
        self.linear_f = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=1, groups=config.embedding_dim, bias=False)
        self.linear_h = nn.Conv1d(config.embedding_dim, config.embedding_dim, kernel_size=1, groups=config.embedding_dim, bias=False)

        # self.outnorm = MultiHeadLayerNorm(ndim=config.embedding_dim, weight=True, bias=config.bias)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

        self.reset_parameters()

    def forward(self, x_t: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = x_t.shape

        x_t = einops.rearrange(x_t, "b s d -> b d s")  # Reshape for Conv1d

        f_gate = self.linear_f(x_t)
        i_gate = self.linear_i(x_t)
        hidden = self.linear_h(x_t)

        f_gate = einops.rearrange(f_gate, "b d s -> b s d")
        i_gate = einops.rearrange(i_gate, "b d s -> b s d")
        hidden = einops.rearrange(hidden, "b d s -> b s d")

        diff = F.softplus(-f_gate) - F.softplus(-i_gate)
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = None
        log_tilde_h = log_g(hidden)

        log_values = log_i + log_tilde_h
        log_coeffs = log_f
        
        h_t = parallel_scan_log(log_coeffs, log_values)
        out = h_t[:, -S:]

        return out

    def reset_parameters(self):
        # forget gate initialization
        torch.nn.init.zeros_(self.linear_i.weight)
        # input gate initialization
        torch.nn.init.zeros_(self.linear_f.weight)
        torch.nn.init.zeros_(self.linear_h.weight)
