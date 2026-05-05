"""
Funkcje treningowe dla DQN (implementacja z tutoriala PyTorch — niezmieniona).
Osobny plik pozwala importować train_dqn() zarówno z notatnika jak i ze skryptu wizualizacji.
"""

import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train_dqn(model_class, n_episodes=600,
              BATCH_SIZE=256, GAMMA=0.99,
              EPS_START=0.9, EPS_END=0.05, EPS_DECAY=500,
              TAU=0.01, LR=5e-4,
              device=None):
    """
    Trenuje agenta DQN i zwraca listę czasów trwania epizodów.

    Parametry:
        model_class  -- klasa sieci (DQN_128x128 lub DQN_128x64)
        n_episodes   -- liczba epizodów treningowych
        BATCH_SIZE   -- rozmiar mini-batcha z replay buffera
        GAMMA        -- współczynnik dyskontowy
        EPS_START    -- początkowy epsilon (eksploracja)
        EPS_END      -- minimalny epsilon
        EPS_DECAY    -- szybkość wygaszania epsilon (w krokach)
        TAU          -- współczynnik soft update target network
        LR           -- learning rate (Adam W)
        device       -- torch.device (autodetect jeśli None)

    Zwraca:
        (policy_net, episode_durations)
    """
    import gymnasium as gym

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

    env = gym.make("CartPole-v1")
    n_actions      = env.action_space.n
    state, _       = env.reset()
    n_observations = len(state)

    policy_net = model_class(n_observations, n_actions).to(device)
    target_net = model_class(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory    = ReplayMemory(10_000)

    steps_done      = 0
    episode_durations = []

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]],
                                device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = \
                target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    for i_episode in range(n_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_t = torch.tensor([reward], device=device)
            done     = terminated or truncated

            next_state = None if terminated else \
                torch.tensor(observation, dtype=torch.float32,
                              device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_t)
            state = next_state

            optimize_model()

            # Soft update target network
            target_net_state_dict  = target_net.state_dict()
            policy_net_state_dict  = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    policy_net_state_dict[key] * TAU
                    + target_net_state_dict[key] * (1 - TAU)
                )
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

    env.close()
    return policy_net, episode_durations


def discounted_total(durations, gamma, max_episodes=1000):
    """
    Całkowita zdyskontowana nagroda z pierwszych max_episodes epizodów.
    Używana jako metryka przy przeszukiwaniu hiperparametrów.

    W CartPole nagroda za epizod = czas trwania (każdy krok = +1 pkt),
    więc dyskontujemy sumy epizodów, nie pojedyncze kroki.
    """
    total = 0.0
    for i, d in enumerate(durations[:max_episodes]):
        total += (gamma ** i) * d
    return total
