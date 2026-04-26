import numpy as np


class CartPoleQLearningAgent:
    """
    Agent Q-learning z dyskretyzacją przestrzeni stanów.

    CartPole-v1 ma ciągłą przestrzeń obserwacji (4 zmienne rzeczywiste),
    którą musimy zamienić na skończoną siatkę indeksów, zanim użyjemy
    tablicy Q.  Im gęstsza siatka, tym dokładniejsza aproksymacja,
    ale też więcej stanów do eksplorowania.
    """

    def __init__(
        self,
        n_actions,
        n_bins=10,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_bins = n_bins

        # Przedziały dyskretyzacji — clipping wartości skrajnych do rozsądnych granic.
        self.bins = [
            np.linspace(-4.8,  4.8,  n_bins - 1),   # pozycja wózka
            np.linspace(-4.0,  4.0,  n_bins - 1),   # prędkość wózka
            np.linspace(-0.418, 0.418, n_bins - 1), # kąt kija [rad]
            np.linspace(-4.0,  4.0,  n_bins - 1),   # prędkość kątowa kija
        ]

        self.q_table = np.zeros((n_bins,) * 4 + (n_actions,))

    def _discretize(self, state):
        """Zamienia ciągły stan na krotkę indeksów tablicy Q."""
        indices = []
        for val, bins in zip(state, self.bins):
            idx = int(np.digitize(val, bins))
            idx = np.clip(idx, 0, self.n_bins - 1)
            indices.append(idx)
        return tuple(indices)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[self._discretize(state)]))

    def update(self, state, action, reward, next_state, terminated):
        s  = self._discretize(state)
        s_ = self._discretize(next_state)

        best_next = np.max(self.q_table[s_])
        td_target = reward + self.gamma * best_next * (1 - int(terminated))
        self.q_table[s][action] += self.lr * (td_target - self.q_table[s][action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
