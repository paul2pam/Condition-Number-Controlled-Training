import os

from setuptools import find_packages, setup

# Check for uv environment usage
if "VIRTUAL_ENV" not in os.environ:
    print(
        "WARNING: It is recommended to run this installation inside the 'xqc' uv virtual environment."
    )

setup(
    name="xqc",
    version="0.1.0",
    description="XQC: Well-conditioned Optimization Accelerates Deep Reinforcement Learning",
    author="Daniel Palenicek, Florian Vogt",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Dependencies are managed via pyproject.toml
    ],
)
