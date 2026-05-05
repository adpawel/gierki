import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        
        # Zmienione parametry dla wygaszania
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, terminated):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - int(terminated))
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        
    def decay_epsilon(self):
        # Mnożymy epsilon przez decay, ale nie pozwalamy mu spaść poniżej wartości minimalnej
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)