import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_scan_log(log_coeffs, log_values):
    """
    log_coeffs: (batch_size, seq_len, input_size)
    log_values: (batch_size, seq_len + 1, input_size)
    """
    eps = 1e-8
    a_star = torch.cumsum(log_coeffs + eps, dim=1)
    # print(f"a_star shape: {a_star.shape}")
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)


def log_g(x):
    eps = 1e-8
    return torch.where(x >= 0, (F.relu(x)+0.5 + eps).log(), -F.softplus(-x) + eps)


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LowRankFFN(nn.Module):
    def __init__(self, dim: int, rank: int, bias: bool = False, act=True):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=bias)
        self.up = nn.Linear(rank, dim, bias=bias)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):           # (B,S,D)
        return self.up(self.act(self.down(x)))

    def reset_parameters(self):
        small_init_(self.down.weight, dim=self.down.in_features)
        small_init_(self.up.weight, dim=self.up.in_features)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)


class DPLRLinear(nn.Module):
    def __init__(self, dim: int, rank: int, bias: bool = False):
        super().__init__()
        self.d = nn.Parameter(torch.ones(dim))
        self.U = nn.Linear(dim, rank, bias=bias)
        self.V = nn.Linear(rank, dim, bias=bias)

    def forward(self, x):  # (B,S,D)
        return x * self.d + self.V(F.silu(self.U(x)))

    def reset_parameters(self):
        nn.init.ones_(self.d)
        small_init_(self.U.weight, dim=self.U.in_features)
        small_init_(self.V.weight, dim=self.V.in_features)
        if self.U.bias is not None:
            nn.init.zeros_(self.U.bias)
        if self.V.bias is not None:
            nn.init.zeros_(self.V.bias)


class LoRAAroundShared(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)     # shared
        self.Ai = nn.Linear(dim, rank, bias=False)
        self.Bi = nn.Linear(rank, dim, bias=False)
        self.Af = nn.Linear(dim, rank, bias=False)
        self.Bf = nn.Linear(rank, dim, bias=False)
        self.Ah = nn.Linear(dim, rank, bias=False)
        self.Bh = nn.Linear(rank, dim, bias=False)

    def forward_i(self, x): return self.W(x) + self.Bi(F.silu(self.Ai(x)))
    def forward_f(self, x): return self.W(x) + self.Bf(F.silu(self.Af(x)))
    def forward_h(self, x): return self.W(x) + self.Bh(F.silu(self.Ah(x)))

    def reset_parameters(self):
        small_init_(self.W.weight, dim=self.W.in_features)
        small_init_(self.Ai.weight, dim=self.Ai.in_features)
        small_init_(self.Bi.weight, dim=self.Bi.in_features)
        small_init_(self.Af.weight, dim=self.Af.in_features)
        small_init_(self.Bf.weight, dim=self.Bf.in_features)
        small_init_(self.Ah.weight, dim=self.Ah.in_features)
        small_init_(self.Bh.weight, dim=self.Bh.in_features)
        if self.Ai.bias is not None:
            nn.init.zeros_(self.Ai.bias)
        if self.Bi.bias is not None:
            nn.init.zeros_(self.Bi.bias)
        if self.Af.bias is not None:
            nn.init.zeros_(self.Af.bias)
        if self.Bf.bias is not None:
            nn.init.zeros_(self.Bf.bias)
        if self.Ah.bias is not None:
            nn.init.zeros_(self.Ah.bias)
        if self.Bh.bias is not None:
            nn.init.zeros_(self.Bh.bias)


class BottleneckMLP(nn.Module):
    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.down = nn.Linear(dim, inner, bias=False)
        self.mix = nn.Linear(inner, inner, bias=False)  # optional
        self.up = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        z = F.silu(self.down(x))
        z = F.silu(self.mix(z)) if self.mix.weight is not None else z
        return self.up(z)

    def reset_parameters(self):
        small_init_(self.down.weight, dim=self.down.in_features)
        small_init_(self.up.weight, dim=self.up.in_features)
        if self.mix.weight is not None:
            small_init_(self.mix.weight, dim=self.mix.in_features)


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
        small_init_(self.linear1.weight, dim=self.linear1.in_features)
        small_init_(self.linear2.weight, dim=self.linear2.in_features)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)


class minLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.linear_i = nn.Linear(input_dim, hidden_dim, bias=False)
        # self.linear_f = nn.Linear(input_dim, hidden_dim, bias=False)
        # self.linear_h = nn.Linear(input_dim, hidden_dim, bias=False)

        # self.mixer = LoRAAroundShared(hidden_dim, rank=4)
        # self.linear_h = FeedForward(hidden_dim, hidden_dim//32)
        # self.linear_i = FeedForward(hidden_dim, hidden_dim//32)
        # self.linear_f = FeedForward(hidden_dim, hidden_dim//32)

        self.linear_i = nn.Conv1d(
            input_dim, hidden_dim, kernel_size=1, groups=128, bias=False)
        self.linear_f = nn.Conv1d(
            input_dim, hidden_dim, kernel_size=1, groups=128, bias=False)
        self.linear_h = nn.Conv1d(
            input_dim, hidden_dim, kernel_size=1, groups=128, bias=False)

    def forward(self, x_t, pre_h=None):
        """
        x_t: (batch_size, sequence_length, input_size)
        """
        B, S, _ = x_t.shape

        x_t = einops.rearrange(x_t, "b s d -> b d s")  # Reshape for Conv1d

        f_gate = self.linear_f(x_t)
        i_gate = self.linear_i(x_t)
        hidden = self.linear_h(x_t)

        # i_gate = self.mixer.forward_i(x_t)
        # f_gate = self.mixer.forward_f(x_t)
        # hidden = self.mixer.forward_h(x_t)

        f_gate = einops.rearrange(f_gate, "b d s -> b s d")
        i_gate = einops.rearrange(i_gate, "b d s -> b s d")
        hidden = einops.rearrange(hidden, "b d s -> b s d")

        diff = F.softplus(-f_gate) - F.softplus(-i_gate)
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(pre_h) if pre_h is not None else None
        log_tilde_h = log_g(hidden)

        if log_h_0 is not None:
            log_values = torch.cat([log_h_0, log_i + log_tilde_h], dim=1)
            log_coeffs = F.pad(log_f, (0, 0, 1, 0))
        else:
            log_values = log_i + log_tilde_h
            log_coeffs = log_f

        h_t = parallel_scan_log(log_coeffs, log_values)
        out = h_t[:, -S:]

        return out

    def reset_parameters(self):
        # small_init_(self.linear_f.weight, dim=self.hidden_dim)
        # small_init_(self.linear_i.weight, dim=self.hidden_dim)
        # small_init_(self.linear_h.weight, dim=self.hidden_dim)

        self.linear_f.reset_parameters()
        self.linear_i.reset_parameters()
        self.linear_h.reset_parameters()

        # self.mixer.reset_parameters()


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.alpha, self.alpha_init_value)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class minLSTM(nn.Module):
    def __init__(
        self,
        dim,
        direction,
        conv_kind="2d",
        conv_kernel_size=4,
        expansion=2,
        conv_bias=True,
        seqlens=None,
        proj_bias=True,
        init_weights="original",
        norm_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.init_weights = init_weights

        inner_dim = expansion * dim

        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )

        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.cell = minLSTMCell(inner_dim, inner_dim)

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )

        self.reset_parameters()

    def forward(self, x):
        B, S, _ = x.shape

        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        x_inner = self.proj_up(x)
        x_minlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        x_minlstm_conv = self.conv(x_minlstm)
        x_minlstm_conv_act = F.silu(x_minlstm_conv)

        h = torch.zeros(B, self.cell.hidden_dim, device=x.device)
        # print(f"x_minlstm_conv_act shape: {x_minlstm_conv_act.shape}")
        # print(f"x shape: {x.shape}")
        # print(f"h shape: {h.shape}")

        h = self.cell(x_minlstm_conv_act)[:, -1:]
        h = h + (self.learnable_skip * x_minlstm_conv_act)
        h = h * F.silu(z)

        # print(h.shape)

        x = self.proj_down(h)

        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x  # (batch_size, seq_len, hidden_dim)

    def reset_parameters(self):
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        if self.init_weights == "original":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":
            wang_init_(self.proj_down.weight, dim=self.dim,
                       num_blocks=self.num_blocks)
        else:
            raise NotImplementedError
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)
        self.cell.reset_parameters()


class minLSTMBlcok(nn.Module):
    def __init__(
        self,
        dim,
        direction,
        drop_path=0.0,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=None,
        init_weights="original",
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights

        self.drop_path = DropPath(drop_prob=drop_path)
        # self.norm = DynamicTanh(normalized_shape=dim, channels_last=True)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = minLSTM(
            dim,
            direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
        )

        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


class minLSTMPair(nn.Module):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=None,
        init_weights="original",
    ):
        super().__init__()
        self.rowwise_from_top_left = minLSTMBlcok(
            dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
        )
        self.rowwise_from_bot_right = minLSTMBlcok(
            dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
        )

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x


class VisionMinLSTM(nn.Module):
    def __init__(
        self,
        dim=192,
        input_shape=(3, 224, 224),
        patch_size=16,
        depth=12,
        output_shape=(1000,),
        mode="classifier",
        pooling="bilateral_flatten",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        legacy_norm=False,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.mode = mode
        self.pooling = pooling

        # Initialize patch embedding
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=input_shape[0],
            resolution=input_shape[1:],
            patch_size=patch_size,
        )

        # Positional embedding
        self.pos_embed = VitPosEmbed2d(
            seqlens=self.patch_embed.seqlens, dim=dim)

        # Stacked minLSTM layers
        self.layers = nn.ModuleList([minLSTMPair(dim) for _ in range(depth)])

        # Normalization
        # self.norm = DynamicTanh(normalized_shape=dim, channels_last=True)
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)

        if pooling == "bilateral_flatten" and mode == "classifier":
            head_dim = dim * 2
        else:
            head_dim = dim

        if legacy_norm:
            # self.legacy_norm = DynamicTanh(
            #     normalized_shape=head_dim, channels_last=True)
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()

        # Classification head
        if mode == "classifier":
            self.head = nn.Linear(head_dim, output_shape[0])
        else:
            self.head = None

    def forward(self, x):
        x = self.patch_embed(x)
        # print(f"After patch_embed - min: {x.min().item()}, max: {x.max().item()}")
        # if torch.isnan(x).any():
        #     print("NaN detected after patch_embed!")
        # add pos_embed
        x = self.pos_embed(x)
        # print(f"After pos_embed - min: {x.min().item()}, max: {x.max().item()}")
        # if torch.isnan(x).any():
        #     print("NaN detected after pos_embed!")

        # print("The shape of the input after positional embedding is: ", x.shape)

        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # Process through minLSTM layersse
        for layer in self.layers:
            # print("The layer is: ", layer)
            x = layer(x)  # (B, S, dim)
        x = self.norm(x)  # Add residual connection and normalize

        # Pooling
        if self.pooling == "bilateral_flatten":
            x = torch.cat([x[:, 0], x[:, -1]], dim=1)
            x = self.legacy_norm(x)
        elif self.pooling == "bilateral_avg":
            x = (x[:, 0] + x[:, -1]) / 2
        elif self.pooling == "to_image":
            x = einops.rearrange(
                x, "b (h w) c -> b c h w", h=self.patch_embed.seqlens[0], w=self.patch_embed.seqlens[1])
        else:
            raise NotImplementedError(
                f"Pooling mode '{self.pooling}' is not implemented")

        # Classification head
        if self.head is not None:
            x = self.head(x)

        return x
