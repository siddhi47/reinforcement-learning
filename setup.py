from setuptools import setup, find_packages

setup(
    name="reinforcement-learning-toolkit",
    version="0.1.0",
    description="A toolkit for reinforcement learning",
    packages=find_packages(),
    requires=[
        "numpy",
    ],
)
