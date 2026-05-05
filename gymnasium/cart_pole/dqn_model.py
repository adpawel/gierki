import torch.nn as nn
import torch.nn.functional as F


class DQN_128x128(nn.Module):
    """
    Konfiguracja 1: dwie warstwy ukryte po 128 jednostek.
    Większa pojemność — lepiej aproksymuje złożone funkcje Q,
    ale wymaga więcej danych i jest wolniejsza.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN_128x128, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN_128x64(nn.Module):
    """
    Konfiguracja 2: pierwsza warstwa 128, druga 64 jednostki.
    Mniejsza pojemność — szybsza, mniej podatna na przeuczenie,
    ale może nie wystarczyć dla bardziej złożonych środowisk.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN_128x64, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
