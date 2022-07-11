from researchpractice.commons import SinusoidalConvPosEmb
import torch


def test_SinusoidalConvPosEmb():
    x = torch.randn(10, 36, 64, 64)
    model = SinusoidalConvPosEmb()
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test_SinusoidalConvPosEmb()
