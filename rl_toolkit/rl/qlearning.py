import numpy as np

class RL:
    def __init__(self, mdp):
        self.mdp = mdp



    def sample_reward_next_state(self, state, action):
        print("state: ", state, "action: ", action)
        print("R: ", self.mdp.R)
        reward = np.random.normal(self.mdp.R[ state, action], 1)

        cumProb = np.cumsum(self.mdp.T[state, action, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]
    
    def select_action(self, state, Q, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.mdp.n_actions)
        else:
            action = np.argmax(Q[state::])
        return action



    def q_learning(self, s0, initial_q, n_episodes, n_steps, epsilon):
        Q = initial_q
        for _ in range(n_episodes):
            state = s0
            for __ in range(n_steps):
                action = self.select_action(state, Q, epsilon)
                [reward, nextState] = self.sample_reward_next_state(state, action)
                Q[action, state] = Q[action, state] + 0.1 * (reward + np.max(Q[nextState, :]) - Q[action, state])
                state = nextState
        return Q


