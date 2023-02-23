"""
    author: siddhi47
    description: simple MDP
    created: 02/17/2023
"""

import numpy as np


class MDP:
    def __init__(self, T: np.ndarray, R: np.ndarray, discount: float):
        """
        T: |S|x|A|x|S'| transition probability matrix
        R: |A|X|S| array
        discount: discount factor(gamma) between 0-1
        """
        if not (type(T) == type(R) == np.ndarray):
            raise TypeError("T and R must be numpy arrays")
        if not (T.ndim == 3 and R.ndim == 2):
            raise ValueError("T and R must be 3D and 2D arrays respectively")
        if type(discount) is not float:
            raise TypeError("discount must be a float")
        if not (0 <= discount < 1):
            raise ValueError("discount must be between 0 and 1, i.e. [0,1]")
        self.T = T
        self.R = R
        self.discount = discount

    def value_iteration(
        self, initial_v: np.ndarray, n_iteration=np.inf, tolerance=0.01
    ) -> np.ndarray:
        """
        params
            initial_v: |S| array
            n_iteration: number of iterations
            tolerance: tolerance for convergence

        returns:
            V: |S| array of values
        """
        epsilon = np.inf
        # to count the number of epochs.
        counter = 0
        while True:
            if counter == 0:
                old_v = initial_v

            # bellmans equation
            new_v: np.ndarray = self.R + self.discount * np.dot(self.T, old_v)
            max_v = np.maximum(*new_v)
            print("Step: ", counter, "\nMax V: ", max_v)
            counter -= -1
            epsilon = np.linalg.norm(max_v - old_v)
            old_v = max_v

            # loop till convergence or max iterations
            if counter > n_iteration or epsilon < tolerance:
                break
        print("Converged in ", counter, " steps")
        return new_v

    def extract_policy(self, V: np.ndarray) -> np.ndarray:
        """
        prams:
            V: |S| array of  values

        returns:
            policy: |S| array of actions
        """
        return np.argmax(V, axis=0)

    def evaluate_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        params:
            policy: |S| array of policies

        returns:
            optimal value for policy
        """
        T = self.select_T_from_policy(policy)
        # computing for initial value
        # V = inv(1-gamma*P)R

        # evaluation
        I = np.identity(self.T[0].shape[0])
        M = I - self.discount * T  # choosing a random polcicy
        M_inv = np.linalg.inv(M)
        V0 = np.dot(M_inv, self.R)
        return V0

    def select_T_from_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        params:
            policy: |S| array of policies

        returns:
            T: |S|x|S'| transition probability matrix
        """

        T = []
        for i, p in enumerate(policy):
            T.append(self.T[p[0]][i])
        return np.array(T)

    def policy_iteration_using_linalg(
        self, initial_policy: np.ndarray, n_iteration=np.inf
    ) -> np.ndarray:
        """
        params:
            initial_policy: Initial policy: array of |S| entries
            n_iteration: number of iteration

        return:
            new_policy of |S| entries
        """
        T = self.select_T_from_policy(initial_policy)
        # computing for initial value
        # V = inv(1-gamma*P)R

        # evaluation
        I = np.identity(self.T[0].shape[0])
        M = I - self.discount * T  # choosing a random polcicy
        M_inv = np.linalg.inv(M)
        V0 = np.dot(M_inv, self.R)
        # improve
        V = self.R + self.discount * np.dot(self.T, V0)
        policy = np.argmax(V, axis=0)
        print("Value: ", V)
        h = 0
        updated_policy = policy
        while h < n_iteration:
            T = self.select_T_from_policy(updated_policy)
            M = I - self.discount * T
            M_inv = np.linalg.inv(M)
            Vi = np.dot(M_inv, self.R)
            print(f"Value{h}: ", Vi)
            # improve
            V = self.R + self.discount * np.dot(self.T, Vi)
            new_policy = np.argmax(V, axis=0)
            if all(np.equal(updated_policy, new_policy)):
                break
            updated_policy = new_policy
            h -= -1
        return updated_policy

    def policy_iteration(
        self, initial_policy: np.ndarray, n_iteration=np.inf
    ) -> np.ndarray:
        """
        params:
            initial_policy: Initial policy: array of |S| entries
            n_iteration: number of iteration

        return:
            new_policy of |S| entries
        """
        # evaluate
        V0 = self.evaluate_policy(
            initial_policy,
        )

        # improve
        V = self.R + self.discount * np.dot(self.T, V0)
        policy = np.argmax(V, axis=0)
        print("Value 0: \n", V)

        h = 0
        updated_policy = policy
        while h < n_iteration:
            # evaluate
            Vi = self.evaluate_policy_partially(
                updated_policy, np.array([[0], [0], [0], [0]])
            )

            # improve
            V = self.R + self.discount * np.dot(self.T, Vi)
            print(f"Value {h+1}: \n", V)
            print(f"Optimal Value {h+1}: \n", np.maximum(*V))
            new_policy = np.argmax(V, axis=0)
            if all(np.equal(updated_policy, new_policy)):
                break
            updated_policy = new_policy
            h -= -1

        print(f"Converged in {h+1} steps")
        return updated_policy

    def evaluate_policy_partially(
        self,
        policy: np.ndarray,
        initial_v: np.ndarray,
        n_iteration=np.inf,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        params:
            policy : Policy: array of |S| entries
            initial_v: Initial value function: array of |S| entries
            n_iterations: limit on the number of iterations: scalar (default: infinity)
            tolerance: threshold on ‖𝑉𝑛 − 𝑉𝑛+1‖∞ that will be compared to a variable epsilon
            (initialized to np.inf): scalar (default: 0.01)

        returns:

            New value function: array of |S| entries.
        """
        counter = 0
        while True:
            if counter == 0:
                old_v = initial_v

            T = self.select_T_from_policy(policy)
            # bellmans equation
            new_v: np.ndarray = self.R + self.discount * np.dot(T, old_v)
            counter -= -1
            epsilon = np.linalg.norm(new_v - old_v)
            old_v = new_v

            # loop till convergence or max iterations
            if counter > n_iteration or epsilon < tolerance:
                break

        return new_v

    def modified_policy_iteration(
        self,
        initial_policy: np.ndarray,
        n_eval_iteration=5,
        n_iteration=np.inf,
        tolerance=0.01,
    ) -> np.ndarray:
        """

        a procedure for the modified policy iteration def modifiedPolicyIteration () that

        params
        initial_policy – Initial policy: array of |S| entries
        initial_v -- Initial value function: array of |S| entries
        n_eval_iterations -- limit on the number of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        n_iterations -- limit on the number of iterations to be performed in modified policy iteration: scalar (default: infinity)
        tolerance -- threshold on ‖𝑉𝑛 − 𝑉𝑛+1‖∞ that will be compared to a variable epsilon(initialized to np.inf): scalar (default: 0.01)
        """
        V0 = self.evaluate_policy_partially(
            initial_policy, np.array([[0], [0], [0], [0]]), n_eval_iteration
        )
        # improve
        V = self.R + self.discount * np.dot(self.T, V0)
        policy = np.argmax(V, axis=0)
        print("Initial Policy: \n", initial_policy)
        print(f"P0:\n {policy}")
        print("Value0: \n", V0)
        h = 0
        updated_policy = policy
        while h < n_iteration:
            # evaluate
            Vi = self.evaluate_policy_partially(
                updated_policy, np.array(
                    [[0], [0], [0], [0]]), n_eval_iteration
            )
            print(f"Value{h+1}: \n", Vi)
            # improve
            V = self.R + self.discount * np.dot(self.T, Vi)
            new_policy = np.argmax(V, axis=0)
            print(f"P{h+1}:\n {new_policy}")
            if all(np.equal(updated_policy, new_policy)):
                break
            updated_policy = new_policy
            h -= -1
        return updated_policy
