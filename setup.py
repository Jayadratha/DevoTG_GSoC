#!/usr/bin/env python3
"""Setup script for DevoTG package."""

from setuptools import setup, find_packages

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)
        return requirements

setup(
    name="devotg",
    version="0.1.0",
    author="Jayadratha Gayen",
    author_email="contactjayag@gmail.com",
    description="Developmental Temporal Graph Networks for C. elegans Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jayadratha/DevoTG",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.12",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "gpu": [
            "torch[cuda]",
        ],
        "colab": [
            "google-colab",
        ]
    },
    entry_points={
        "console_scripts": [
            "devotg-analyze=scripts.run_analysis:main",
            "devotg-train=scripts.train_tgn:main",
            "devotg-visualize=scripts.generate_visualizations:main",
        ],
    },
    include_package_data=True,
    package_data={
        "devotg": [
            "config/*.yaml",
            "data/sample/*.csv",
        ],
    },
    zip_safe=False,
    keywords="temporal-graphs neural-networks developmental-biology c-elegans visualization",
    project_urls={
        "Bug Reports": "https://github.com/Jayadratha/DevoTG/issues",
        "Source": "https://github.com/Jayadratha/DevoTG",
        "Documentation": "https://github.com/Jayadratha/DevoTG#readme",
    },
)