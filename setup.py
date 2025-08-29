#!/usr/bin/env python3
"""
Setup script for the DevoTG package.
run via: python setup2.py install
or: python setup2.py bdist_wheel
or: python setup2.py sdist
or: python setup2.py bdist_wheel sdist

This script handles the packaging and distribution of the DevoTG library,
which provides tools for analyzing and visualizing temporal graph data,
with a specific focus on developmental biology datasets like the C. elegans connectome.
"""

import os
from setuptools import setup, find_packages

def read_readme():
    """Read the README file for the long description."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

def read_requirements():
    """
    Return pip-installable requirements for DevoTG.
    Avoids conda/system packages and focuses on core dependencies.
    """
    return [
        # Core scientific stack
        "numpy>=2.0",
        "scipy>=1.15",
        "pandas>=2.2",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "scikit-learn>=1.5",
        "plotly>=5.20",
        "imageio>=2.35",

        # Graph & torch ecosystem
        "torch>=2.5",
        "torch-geometric>=2.6",
        "torch-geometric-temporal>=0.56",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "networkx>=3.3",

        # Other utilities
        "tqdm>=4.66",
        "typing-extensions>=4.12",
        "openpyxl>=3.1",
        "aiohttp>=3.12",
        "jinja2>=3.1",
        "fsspec>=2024.6",
        "requests>=2.32",
    ]

setup(
    name="devotg",
    version="0.3.0",
    author="Jayadratha Gayen",
    author_email="jayadratha.gayen@research.iiit.ac.in",
    license="MIT",
    description="A Python framework for analyzing and visualizing developmental temporal graphs, demonstrated on the C. elegans connectome.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jayadratha/DevoTG_GSoC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires="~=3.12",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "isort>=5.10",
            "jupyterlab>=3.0",
            "notebook>=6.4",
        ],
        "gpu": [
            "torch-geometric[cuda]",
            "torch[cuda]",
        ],
    },
    entry_points={
        "console_scripts": [
            "devotg-connectome-analysis=scripts.run_connectome_analysis:main",
            "devotg-cell-lineage-analysis=scripts.run_cell_lineage_analysis:main",
            "devotg-train-tgn=scripts.train_tgn:main",
        ],
    },
    include_package_data=True,
    package_data={
        "devotg": ["config.yaml"],
    },
    zip_safe=False,
    keywords="temporal-graphs, developmental-biology, c-elegans, network-analysis, visualization, pytorch-geometric",
    project_urls={
        "Bug Tracker": "https://github.com/Jayadratha/DevoTG_GSoC/issues",
        "Documentation": "https://github.com/Jayadratha/DevoTG_GSoC/blob/main/README.md",
        "Source Code": "https://github.com/Jayadratha/DevoTG_GSoC",
    },
)