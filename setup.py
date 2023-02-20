from setuptools import setup, find_packages

setup(
    name='markov-decision-process',
    version='0.1.0',
    description='Markov Decision Process',
    full_description='Markov Decision Process using policy iteration, value iteration and modified policy iteration',
    packages=find_packages(),
    requires=['numpy',]
)
