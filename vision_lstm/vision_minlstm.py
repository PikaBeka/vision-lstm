import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d

def parallel_scan_log(log_coeffs, log_values):
    """
    log_coeffs: (batch_size, seq_len, input_size)
    log_values: (batch_size, seq_len + 1, input_size)
    """
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 0, 1))
    # print(f"a_star shape: {a_star.shape}")
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))

class minLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_f = nn.Linear(input_dim, hidden_dim)
        self.linear_i = nn.Linear(input_dim, hidden_dim)
        self.linear_h = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x_t, pre_h):
        """
        x_t: (batch_size, sequence_length, input_size)
        pre_h: (batch_size, 1, hidden_size)
        """
        diff = F.softplus(-self.linear_f(x_t)) - F.softplus(-self.linear_i(x_t))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(pre_h)
        log_tilde_h = log_g(self.linear_h(x_t))
        
        # Ensure consistent dimensions for `log_values`
        log_values = torch.cat([log_h_0, log_i + log_tilde_h], dim=1)
        # print(f"log_f shape: {log_f.shape}")
        # print(f"log_values shape: {log_values.shape}")
        h = parallel_scan_log(log_f, log_values)
        return h

class minLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = minLSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        h = torch.zeros(x.size(0), 1, self.cell.hidden_dim, device=x.device)
        outputs = []
        for t in range(x.size(1)):
            h = self.cell(x[:, t:t+1, :], h)
            outputs.append(h)
        return torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)

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

    def forward(self):
        pass

class minLSTMPair(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
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
        self.rowwise_from_top_left = ViLBlock()
        self.rowwise_from_bot_right = ViLBlock()
    
    def forward(self):
        pass

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
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # Stacked minLSTM layers
        self.layers = nn.ModuleList([minLSTM(dim, dim) for _ in range(depth)])

        # Normalization
        self.norm = nn.LayerNorm(dim)

        if pooling == "bilateral_flatten" and mode == "classifier":
            head_dim = dim * 2
        else:
            head_dim = dim

        # Classification head
        if mode == "classifier":
            self.head = nn.Linear(head_dim, output_shape[0])
        else:
            self.head = None

    def forward(self, x):
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        # print("The shape of the input after positional embedding is: ", x.shape)

        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # Process through minLSTM layersse
        for layer in self.layers:
            # print("The layer is: ", layer)
            x_residual = x
            x = layer(x)  # (B, S, dim)
            x = self.norm(x + x_residual)  # Add residual connection and normalize

        # Pooling
        if self.pooling == "bilateral_flatten":
            x = torch.cat([x[:, 0], x[:, -1]], dim=1)
        elif self.pooling == "bilateral_avg":
            x = (x[:, 0] + x[:, -1]) / 2
        elif self.pooling == "to_image":
            x = einops.rearrange(x, "b (h w) c -> b c h w", h=self.patch_embed.seqlens[0], w=self.patch_embed.seqlens[1])
        else:
            raise NotImplementedError(f"Pooling mode '{self.pooling}' is not implemented")

        # Classification head
        if self.head is not None:
            x = self.head(x)
        
        return x
