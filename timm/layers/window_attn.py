""" Window Attention

Local window-based multi-head self-attention (W-MSA) with relative position bias,
as an NCHW feature-map layer that can be dropped into existing conv-style models.

Paper: `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/abs/2103.14030

Hacked together by / Copyright 2026 Ross Wightman
"""
from typing import Final, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.nn import functional as F

from ._fx import register_notrace_function
from .config import use_fused_attn
from .helpers import to_2tuple
from .pos_embed_rel import RelPosBias
from .trace_utils import _assert

_int_or_tuple_2_t = Union[int, Tuple[int, int]]

__all__ = ['window_partition', 'window_reverse', 'WindowAttention2d']


def window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int],
) -> torch.Tensor:
    """Partition an NCHW feature map into non-overlapping windows.

    Args:
        x: Input tensor with shape ``(B, C, H, W)``.
        window_size: Window size ``(window_h, window_w)``.

    Returns:
        Windows with shape ``(B * num_windows, C, window_h, window_w)``.
    """
    B, C, H, W = x.shape
    win_h, win_w = window_size
    _assert(H % win_h == 0, f'height ({H}) must be divisible by window height ({win_h})')
    _assert(W % win_w == 0, f'width ({W}) must be divisible by window width ({win_w})')
    x = x.view(B, C, H // win_h, win_h, W // win_w, win_w)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, win_h, win_w)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(
        windows: torch.Tensor,
        window_size: Tuple[int, int],
        H: int,
        W: int,
) -> torch.Tensor:
    """Reverse :func:`window_partition`.

    Args:
        windows: Tensor with shape ``(B * num_windows, C, window_h, window_w)``.
        window_size: Window size ``(window_h, window_w)``.
        H: Height of the original feature map.
        W: Width of the original feature map.

    Returns:
        Feature map with shape ``(B, C, H, W)``.
    """
    C = windows.shape[1]
    win_h, win_w = window_size
    x = windows.view(-1, H // win_h, W // win_w, C, win_h, win_w)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


class WindowAttention2d(nn.Module):
    """Window based multi-head self-attention (W-MSA) for NCHW feature maps.

    The input feature map is partitioned into non-overlapping windows and standard
    multi-head self-attention with a learned relative position bias is computed
    independently within each window. A per-window additive mask can be supplied to
    implement shifted-window (SW-MSA) style attention.

    As with the other ``*2d`` attention layers, the input and output are NCHW tensors
    with the same spatial size, so the module can replace a convolution or another
    local attention block (e.g. :class:`~timm.layers.HaloAttn`) in existing models.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            device=None,
            dtype=None,
    ):
        """
        Args:
            dim: Number of input channels.
            dim_out: Number of output channels, same as ``dim`` if not set.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (``dim_out // num_heads`` if not set).
            window_size: Window size as an int or ``(height, width)`` tuple.
            qkv_bias: If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, 'dim_out must be divisible by num_heads'
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = head_dim or dim_out // num_heads
        attn_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.window_size = to_2tuple(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]

        self.rel_pos = RelPosBias(
            window_size=self.window_size,
            num_heads=num_heads,
            **dd,
        )

        self.qkv = nn.Conv2d(dim, attn_dim * 3, kernel_size=1, bias=qkv_bias, **dd)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(attn_dim, dim_out, kernel_size=1, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_window_size(self, window_size: _int_or_tuple_2_t) -> None:
        """Update window size, resizing the relative position bias table.

        Args:
            window_size: New window size as int or ``(height, width)`` tuple.
        """
        window_size = to_2tuple(window_size)
        if window_size == self.window_size:
            return
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.rel_pos.set_window_size(window_size)

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers (e.g. after loading onto a new device)."""
        self.rel_pos.init_non_persistent_buffers()

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run window attention.

        Args:
            x: Input features with shape ``(B, C, H, W)``. ``H`` and ``W`` must be
                divisible by the configured window size.
            attn_mask: Optional additive float mask with shape
                ``(num_windows, window_area, window_area)``, e.g. the cyclic-shift
                mask used by shifted-window attention.

        Returns:
            Output features with shape ``(B, dim_out, H, W)``.
        """
        B, C, H, W = x.shape
        win_h, win_w = self.window_size
        num_windows = (H // win_h) * (W // win_w)
        N = self.window_area

        # (B * num_windows, 3*attn_dim, win_h, win_w)
        qkv = self.qkv(window_partition(x, self.window_size))
        q, k, v = qkv.reshape(
            qkv.shape[0], 3, self.num_heads, self.head_dim, N
        ).unbind(1)
        # each of q, k, v: (B * num_windows, num_heads, window_area, head_dim)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        rel_pos_bias = self.rel_pos.get_bias().to(q.dtype)  # (1, num_heads, N, N)
        if attn_mask is not None:
            # expand per-window mask over batch and heads
            rel_pos_bias = rel_pos_bias + attn_mask.view(
                1, num_windows, 1, N, N
            ).expand(B, -1, self.num_heads, -1, -1).reshape(
                B * num_windows, self.num_heads, N, N
            )

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=rel_pos_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn + rel_pos_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # (B * num_windows, num_heads * head_dim, win_h, win_w)
        x = x.transpose(1, 2).reshape(-1, self.num_heads * self.head_dim, win_h, win_w)
        x = window_reverse(x, self.window_size, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
