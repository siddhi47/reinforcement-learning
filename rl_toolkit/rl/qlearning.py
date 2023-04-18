import numpy as np

class RL:
    def __init__(self, mdp):
        """
        :param mdp: Markov Decision Process
        """

        self.mdp = mdp

    def sample_reward_next_state(self, state, action):
        """
        Sample a reward and next state from the MDP
        :param state: current state
        :param action: action to take
        :return: reward, next state
        """

        reward = np.random.normal(self.mdp.R[action, state], 1)
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def select_action(self, state, Q, epsilon):
        """
        Select an action using epsilon-greedy policy
        :param state: current state
        :param Q: Q-table
        :param epsilon: probability of selecting a random action
        :return: action
        """

        if epsilon>0:
            # eplison-greedy
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, self.mdp.n_actions)
            else:
                action = np.argmax(Q[:, state])
            return action
        else:
            # greedy
            return np.argmax(Q[:, state])

    def q_learning(self, s0, initial_q, n_episodes, n_steps, epsilon):
        """
        Q-learning algorithm
        :param s0: initial state
        :param initial_q: initial Q-table
        :param n_episodes: number of episodes
        :param n_steps: number of steps per episode
        :param epsilon: probability of selecting a random action
        :return: [Q table, policy]

        """

        Q = initial_q
        rewards = np.zeros(n_episodes)
        n = np.zeros((self.mdp.n_actions, self.mdp.n_states))
        for _ in range(n_episodes):
            state = s0
            for __ in range(n_steps):
                action = self.select_action(state, Q, epsilon)
                reward, nextState = self.sample_reward_next_state(state, action)
                rewards[_] = reward
                n[action, state] += 1
                alpha = 1 / n[action, state]

                Q[action, state] = Q[action, state] + alpha * (
                    reward + np.max(Q[:, nextState]) - Q[action, state]
                )
                Q[action, state] = np.round(Q[action, state], 2)
                state = nextState
        return Q, np.argmax(Q, axis=0)
