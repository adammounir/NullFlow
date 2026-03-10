from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="nullflow",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="NullFlow: Task-Free Continual Learning via Null-Space Constrained Latent Flow Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous-nullflow/NullFlow",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
