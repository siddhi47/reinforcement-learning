import pytest
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_toolkit.rl.qlearning import RL
from tests.test_rl.rl_fixtures import *


def test_qlearning(mdp):
    rl = RL(mdp)
    s0 = 0
    initial_q = np.zeros((mdp.n_actions, mdp.n_states))
    n_episodes = 100
    n_steps = 100
    epsilon = 0.1
    Q = rl.q_learning(s0, initial_q, n_episodes, n_steps, epsilon)
    print(Q)

