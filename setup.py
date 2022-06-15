#!/usr/bin/env python3

from setuptools import find_packages, setup


# Load the current project version
exec(open("blended_tiling/version.py").read())


# Linting & testing tools
DEV_REQUIREMENTS = [
    "black==22.3.0",
    "flake8",
    "mypy>=0.760",
    "pytest",
    "pytest-cov",
    "usort==1.0.2",
    "ufmt",
]


# Use README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="blended-tiling",
    version=__version__,  # type: ignore[name-defined]  # noqa: F821
    license="MIT",
    description="Blended tiling with PyTorch",
    author="Ben Egan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/progamergov/blended-tiling",
    keywords=[
        "blended-tiling",
        "tiler",
        "tiling",
        "masking",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.6",
    ],
    packages=find_packages(exclude=("tests", "tests.*")),
    extras_require={
        "dev": DEV_REQUIREMENTS,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Artistic Software",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
