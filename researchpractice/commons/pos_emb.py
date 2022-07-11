import math
import torch
from torch import nn


class SinusoidalConvPosEmb(nn.Module):
    """Build Conv sinusoidal embeddings.

    Input x with 4 dimensions: (batch_size, n_channels, height, width);
    Output the same 4 dimensions as Input for Position Encoding.

    Refer to
        - Fairseq: https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
        - https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py#L24

    This implementation differs from the description in Section 3.5 of "Attention Is All You Need" and all references in some way.
    It take (h*w) as num of pos along with c as d_model to get position encoder and then reshape back to (c, h, w).
    """

    def __init__(
        self,
    ):
        super(SinusoidalConvPosEmb, self).__init__()

    def forward(self, x):
        n, c, h, w = x.shape
        sinusoidal_pe = SinusoidalConvPosEmb.get_pe(c, h * w)
        sinusoidal_pe = sinusoidal_pe.reshape(c, h, w).unsqueeze(0).repeat(n, 1, 1, 1)
        return sinusoidal_pe

    @staticmethod
    def get_pe(
        emb_dim: int,
        n_pos: int,
    ):
        """Build sinusoidal embeddings for 1d.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = int(emb_dim / 2)
        half_pos = torch.arange(half_dim, dtype=torch.float)
        sinusoidal_in = torch.exp(
            half_pos * -(math.log(10000.0) / (half_dim - 1))
        )  # [half_dim]
        emb = torch.arange(n_pos, dtype=torch.float).unsqueeze(
            1
        ) * sinusoidal_in.unsqueeze(
            0
        )  # [n_pos, half_dim]
        sinusoidal_pe = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            n_pos, -1
        )  # [n_pos, emb_dim]
        if emb_dim % 2 == 1:
            # zero pad
            sinusoidal_pe = torch.cat([sinusoidal_pe, torch.zeros(n_pos, 1)], dim=1)
        return sinusoidal_pe
