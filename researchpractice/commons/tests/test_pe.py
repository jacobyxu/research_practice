import torch
import os
import sys
sys.path.append(os.getcwd())


def test_SinusoidalConvPosEmb():
    from researchpractice.commons import SinusoidalConvPosEmb
    x = torch.randn(10, 36, 64, 64)
    model = SinusoidalConvPosEmb()
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test_SinusoidalConvPosEmb()
