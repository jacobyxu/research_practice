from typing import Optional
import torch
from torch import nn
import einops


class LayerNormChannel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        eps: Optional[float] = 1e-5,
    ):
        super(LayerNormChannel, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        m = torch.mean(x, dim=1, keepdim=True)
        normalized = (x - m) / (var + self.eps).sqrt() * self.g + self.b
        return normalized


class BaseConvQKVAttention(nn.Module):
    """
    Input x with 4 dimensions: (batch_size, n_channels, height, width);
    Output 4 dimensions (1, n_channels, height, width).
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: Optional[int] = 4,
        head_dim: Optional[int] = 32,
        add_layernorm: Optional[bool] = False,
    ):
        super(BaseConvQKVAttention, self).__init__()
        self.scaling = 1 / head_dim ** 0.5
        self.n_heads = n_heads
        self.qkv_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_heads * head_dim * 3,
            kernel_size=1,
            bias=False,
        )
        attn_conv = nn.Conv2d(
            in_channels=n_heads * head_dim,
            out_channels=in_channels,
            kernel_size=1,
        )
        if add_layernorm:
            self.attn_conv = nn.Sequential(attn_conv, LayerNormChannel(in_channels))
        else:
            self.attn_conv = attn_conv

    def split_qkv(self, x):
        """Split Conv block to Q, K, V for any type of QKV Attention."""
        n, c, h, w = x.shape
        assert (
            c % (3 * self.n_heads) == 0
        ), f"n_heads {self.n_heads} is not legal for c = {c}."

        qkv = self.qkv_conv(x).chunk(chunks=3, dim=1)
        # split channels (h * c) to h groups (heads) of c channels and flattened each channel's by x, y into a vector.
        q, k, v = map(
            lambda t: einops.rearrange(t, "n (h c) x y -> n h c (x y)", h=self.n_heads),
            qkv,
        )
        return q, k, v


class ClassicConvQKVAttention(BaseConvQKVAttention):
    """
    QKV attention (transformer) block that allows spatial positions to attend to each other.

    refer to:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/unet.py#L361
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: Optional[int] = 4,
        head_dim: Optional[int] = 32,
        add_layernorm: Optional[bool] = False,
    ):
        super(ClassicConvQKVAttention, self).__init__(
            in_channels, n_heads, head_dim, add_layernorm
        )

    def forward(self, x):
        """Scaled Dot-Product Attention.

        Attention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V
        Paper: https://arxiv.org/pdf/1706.03762.pdf
        """
        n, c, h, w = x.shape
        q, k, v = self.split_qkv(x)

        sim = torch.einsum("n h d i, n h d j -> n h i j", q, k) * self.scaling
        # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L228
        # sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        sim = sim.softmax(dim=-1)

        attn = torch.einsum("n h i d, n h j d -> n h i j", sim, v)
        attn = einops.rearrange(attn, "n h (x y) d -> n (h d) x y", x=h)
        return self.attn_conv(attn)


class EfficientConvQKVAttention(BaseConvQKVAttention):
    """
    Efficient Linear Attention, introduced by [Efficient Attention: Attention with Linear Complexities].
    And More generic formulation is introduced by [Transformers are RNNs:Fast Autoregressive Transformers with Linear Attention].

    refer to:
    https://github.com/cmsflash/efficient-attention/blob/master/efficient_attention.py
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L184
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: Optional[int] = 4,
        head_dim: Optional[int] = 32,
        add_layernorm: Optional[bool] = True,
    ):
        super(EfficientConvQKVAttention, self).__init__(
            in_channels, n_heads, head_dim, add_layernorm
        )

    def forward(self, x):
        """Linear Attention.

        Attention(Q,K,V) = softmax(Q) * ((softmax(K^T) * V) / sqrt(d_k))
        Paper: https://arxiv.org/pdf/1812.01243.pdf
        """
        n, c, h, w = x.shape
        q, k, v = self.split_qkv(x)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        cxt = torch.einsum("n h i d, b h j d -> n h i j", k, v) * self.scaling

        attn = torch.einsum("n h d i, n h d j -> n h i j", cxt, q)
        attn = einops.rearrange(attn, "n h d (x y) -> n (h d) x y", x=h)
        return self.attn_conv(attn)
