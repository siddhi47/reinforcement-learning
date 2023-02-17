"""
    author: siddhi47
    description: simple MDP
    created: 02/17/2023
"""

import numpy as np

class MDP:
    def __init__(self, T:np.ndarray, R:np.ndarray, discount:float):
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
            raise ValueError("discount must be between 0 and 1, i.e. [0,1)")
        self.T = T
        self.R = R
        self.discount = discount

    def value_iteration(self, initial_v:np.ndarray, n_iteration = np.inf, tolerance = 0.01) ->np.ndarray:
        """ 
            initial_v: |S| array
            n_iteration: number of iterations
            tolerance: tolerance for convergence
        """
        counter = 0 
        while True:
            old_v = initial_v
            new_v: np.ndarray = self.R + self.discount*np.dot(self.T,old_v)
            counter -=- 1
            value_diff = np.linalg.norm(new_v-old_v)
            if counter > n_iteration or value_diff<tolerance:
                break

        return new_v

    
    def extract_policy(self, V:np.ndarray)->np.ndarray:
        """
            V: |S| array of 
        """

        return np.array(np.zeros(2))




