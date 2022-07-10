from setuptools import setup, find_packages

setup(
    name="researchpractice",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="Practice Models - Pytorch",
    author="Jacob Xu",
    author_email="",
    url="",
    long_description_content_type="text/markdown",
    keywords=["artificial intelligence"],
    install_requires=[
        # "accelerate",
        "einops",
        # "ema-pytorch",
        "pillow",
        "torch",
        "torchvision",
        "tqdm",
    ],
    classifiers=[
        # "Development Status :: 4 - Beta",
        # "Intended Audience :: Developers",
        # "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # "License :: OSI Approved :: MIT License",
        # "Programming Language :: Python :: 3.6",
    ],
)
