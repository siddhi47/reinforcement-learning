"""
    author:siddhi47
    description: fixtures for MDP test
"""

import pytest
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_toolkit.markov_decision_process.MDP import MDP


@pytest.fixture
def T():
    T = np.array(
        [
            [[0.5, 0.5, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0], [0, 1, 0, 0]],
            [[1, 0, 0, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5]],
        ]
    )

    return T


@pytest.fixture
def R():
    return np.array([[0], [0], [10], [10]])


@pytest.fixture
def gamma():
    return 0.9


@pytest.fixture
def mdp(T, R, gamma):
    return MDP(T, R, gamma)


@pytest.fixture
def initial_v():
    return np.array([[0], [0], [0], [0]])


@pytest.fixture
def initial_policy():
    return np.array([[0], [0], [0], [0]])
