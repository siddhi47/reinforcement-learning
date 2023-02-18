import pytest
import numpy as np
from MDP import MDP

@pytest.fixture
def T():
    T = np.array(
            [
                [[0.5, 0.5, 0, 0],
                [0, 1, 0, 0],
                [0.5, 0.5, 0, 0],
                [0, 1, 0, 0]],

                [[1, 0, 0, 0],
                [0.5, 0, 0, 0.5],
                [0.5, 0, 0.5, 0],
                [0, 0, 0.5, 0.5]]
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

def test_MDP(T, R, gamma):
    MDP(T, R, gamma)

def test_value_iteration(mdp, initial_v):
    value = mdp.value_iteration(initial_v,)
    print("\nValue Function")
    print(value)
    print(30*"*")

def test_evaluate_policy(mdp, initial_v):
    value = mdp.value_iteration(initial_v)
    policy = mdp.evaluate_policy(value)
    print("Policy evaluation")
    print(policy)
    print(30*"*")

def test_extract_policy(mdp, initial_v):
    value = mdp.value_iteration(initial_v)
    action = mdp.extract_policy(value)
    print("Policy extraction")
    print(action)
    print(30*"*")

