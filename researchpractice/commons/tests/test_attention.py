from researchpractice.commons import ClassicConvQKVAttention, EfficientConvQKVAttention
import torch


def test_ClassicConvQKVAttention():
    x = torch.randn(1, 36, 64, 64)
    model = ClassicConvQKVAttention(in_channels=36)
    y = model(x)
    print(x.shape)
    print(y.shape)


def test_LinearConvQKVAttention():
    x = torch.randn(1, 36, 64, 64)
    model = EfficientConvQKVAttention(in_channels=36)
    y = model(x)
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test_ClassicConvQKVAttention()
    test_LinearConvQKVAttention()
