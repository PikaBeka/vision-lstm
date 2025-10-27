# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

import torch
from torch import nn

from ...components.init import small_init_init_, wang_init_
from ...components.ln import MultiHeadLayerNorm
from ...components.dynamic_tanh import DynamicTanh
from .backends import parallel_scan_log, log_g
import einops
import torch.nn.functional as F


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1
    bias: bool = False


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters(self):
        small_init_init_(self.linear1.weight, dim=self.linear1.in_features)
        small_init_init_(self.linear2.weight, dim=self.linear2.in_features)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)


class mLSTMCell(nn.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config

        self.backend_fn = parallel_scan_log

        self.linear_i = nn.Conv1d(config.embedding_dim, config.embedding_dim,
                                  kernel_size=1, groups=128, bias=False)
        self.linear_f = nn.Conv1d(config.embedding_dim, config.embedding_dim,
                                  kernel_size=1, groups=128, bias=False)
        self.linear_h = nn.Conv1d(config.embedding_dim, config.embedding_dim,
                                  kernel_size=1, groups=128, bias=False)

        # self.linear_h = FeedForward(
        #     config.embedding_dim, config.embedding_dim//128)
        # self.linear_i = FeedForward(
        #     config.embedding_dim, config.embedding_dim//128)
        # self.linear_f = FeedForward(
        #     config.embedding_dim, config.embedding_dim//128)

        # self.norm = DynamicTanh(
        #     normalized_shape=config.embedding_dim, channels_last=True)

        self.reset_parameters()

    def forward(self, x_t: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = x_t.shape

        x_t = einops.rearrange(x_t, "b s d -> b d s")  # Reshape for Conv1d

        # x_t_norm = self.norm(x_t)

        f_gate = self.linear_f(x_t)
        i_gate = self.linear_i(x_t)
        hidden = self.linear_h(x_t)

        f_gate = einops.rearrange(f_gate, "b d s -> b s d")
        i_gate = einops.rearrange(i_gate, "b d s -> b s d")
        hidden = einops.rearrange(hidden, "b d s -> b s d")

        diff = F.softplus(-f_gate) - F.softplus(-i_gate)
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_tilde_h = log_g(hidden)

        log_values = log_i + log_tilde_h
        log_coeffs = log_f

        h_t = parallel_scan_log(log_coeffs, log_values)
        out = h_t[:, -S:]

        return out

    def reset_parameters(self):
        torch.nn.init.zeros_(self.linear_i.weight)
        torch.nn.init.zeros_(self.linear_f.weight)
        torch.nn.init.zeros_(self.linear_h.weight)

        # self.linear_f.reset_parameters()
        # self.linear_i.reset_parameters()
        # self.linear_h.reset_parameters()

# # This file is licensed under Apache-2.0
# # Copyright (c) NXAI GmbH and its affiliates 2024
# # Maximilian Beck
# from dataclasses import dataclass

# import torch
# from torch import nn

# from ...components.init import bias_linspace_init_
# from ...components.ln import MultiHeadLayerNorm
# from .backends import parallel_stabilized_simple


# @dataclass
# class mLSTMCellConfig:
#     context_length: int = -1
#     embedding_dim: int = -1
#     num_heads: int = -1
#     bias: bool = False


# class mLSTMCell(nn.Module):
#     config_class = mLSTMCellConfig

#     def __init__(self, config: mLSTMCellConfig):
#         super().__init__()
#         self.config = config

#         self.backend_fn = parallel_stabilized_simple

#         self.igate = nn.Linear(3 * config.embedding_dim, config.num_heads)
#         self.fgate = nn.Linear(3 * config.embedding_dim, config.num_heads)

#         self.outnorm = MultiHeadLayerNorm(ndim=config.embedding_dim, weight=True, bias=config.bias)

#         self.register_buffer(
#             "causal_mask",
#             torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
#             persistent=False,
#         )

#         self.reset_parameters()

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
#         B, S, _ = q.shape  # (B, S, H)

#         if_gate_input = torch.cat([q, k, v], dim=-1)
#         q = q.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
#         k = k.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
#         v = v.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

#         q = q.transpose(1, 2)  # (B, NH, S, DH)
#         k = k.transpose(1, 2)  # (B, NH, S, DH)
#         v = v.transpose(1, 2)  # (B, NH, S, DH)

#         # compute input and forget gate pre-activations
#         igate_preact = self.igate(if_gate_input)  # (B, S, NH)
#         igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
#         fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
#         fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

#         h_state = self.backend_fn(
#             queries=q,
#             keys=k,
#             values=v,
#             igate_preact=igate_preact,
#             fgate_preact=fgate_preact,
#             lower_triangular_matrix=self.causal_mask,
#         )  # (B, NH, S, DH)

#         h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
#         h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

#         return h_state_norm

#     def reset_parameters(self):
#         self.outnorm.reset_parameters()
#         # forget gate initialization
#         torch.nn.init.zeros_(self.fgate.weight)
#         bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
#         # input gate initialization
#         torch.nn.init.zeros_(self.igate.weight)
#         torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
