from setuptools import setup, find_packages

setup(
    name="reinforcement-learning-toolkit",
    version="0.1.0",
    description="A toolkit for reinforcement learning",
    full_description="Includes Markov Decision Process using policy iteration, value iteration and modified policy iteration",
    packages=find_packages(),
    requires=[
        "numpy",
    ],
)
