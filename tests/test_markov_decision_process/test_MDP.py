"""
    author:siddhi47
    description: test for MDP
"""
import pytest
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_toolkit.markov_decision_process.MDP import MDP
import pytest
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl_toolkit.markov_decision_process.MDP import MDP
from tests.test_markov_decision_process.MDP_fixtures import *


def test_MDP(T, R, gamma):
    MDP(T, R, gamma)


def test_value_iteration(mdp, initial_v):
    value = mdp.value_iteration(
        initial_v,
    )
    print("\nValue Function")
    print(value)
    print(30 * "*")


def test_evaluate_policy(mdp, initial_policy):
    policy = mdp.evaluate_policy(initial_policy)
    print("Policy evaluation")
    print(policy)
    print(30 * "*")


def test_extract_policy(mdp, initial_v):
    value = mdp.value_iteration(initial_v)
    action = mdp.extract_policy(value)
    print("Policy extraction")
    print(action)
    print(30 * "*")


def test_policy_iteration(mdp, initial_policy):
    print("Policy Iteration")
    value = mdp.policy_iteration(initial_policy)
    print(value)
    print(30 * "*")


def test_evaluate_policy_partially(mdp, initial_policy, initial_v):
    value = mdp.evaluate_policy_partially(initial_policy, initial_v)

    print("Partial policy evaluation")
    print(value)
    print(30 * "*")


def test_policy_iteration_using_linalg(mdp, initial_policy):
    print("Policy iteration with linear algebra")
    value = mdp.policy_iteration(initial_policy)
    print(value)
    print(30 * "*")


def test_modified_policy_iteration(mdp, initial_policy):
    print("Modified policy iteration")
    value = mdp.modified_policy_iteration(
        initial_policy,
    )
    print(30 * "*")
