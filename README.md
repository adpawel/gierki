# Inteligencja Obliczeniowa w Analizie Danych Cyfrowych

## Projekt 1: Nimby (Gry stochastyczne i algorytmy Minimax)

Projekt implementuje probabilistyczny wariant klasycznej gry matematycznej Nim (tzw. *Nimby*), w którym gracz ma 10% szansy na wzięcie ze stosu o jednego elementu mniej, niż zadeklarował. Głównym celem projektu była analiza wpływu losowości i głębokości przeszukiwania drzewa gry na skuteczność algorytmów decyzyjnych.

### O silniku i Sztucznej Inteligencji
Projekt wykorzystuje framework **`easyAI`**. Wbudowany algorytm `Negamax` wymaga jednak w pełni deterministycznego środowiska (poprawne działanie operacji `make_move` i `unmake_move` podczas budowania drzewa stanów). 

Aby rozwiązać ten problem architektoniczny, symulacje AI wewnątrz drzewa decyzyjnego pozostawiono jako deterministyczne, a czynnik stochastyczny jest ewaluowany *w locie* (in-flight) w głównej pętli gry – tuż po wybraniu optymalnego ruchu przez agenta, a przed jego ostatecznym zaaplikowaniem na planszy. 

W ramach projektu rozszerzono bazowe możliwości biblioteki, implementując własne modele AI:
* **`NegamaxNoAB`** – wariant bez optymalizacji cięć Alfa-Beta, służący do weryfikacji złożoności czasowej (dowodzi spadku wydajności przy zachowaniu tych samych decyzji logicznych).
* **`ExpectiNegamax`** – wariant algorytmu Expectiminimax z odcięciem Alfa-Beta, wprowadzający *węzły szansy* (chance nodes). Oblicza on wartość oczekiwaną ruchu ze wzoru: $E = 0.9 \cdot V_{normal} + 0.1 \cdot V_{slip}$, co pozwala na podejmowanie decyzji świadomych ryzyka.

### Struktura plików

```text
nimby/
├── models.py                     # Własne klasy algorytmów (NegamaxNoAB, ExpectiNegamax)
├── nim.py                        # Klasa bazowa i zasady gry (dziedziczy po easyAI.TwoPlayerGame)
├── utils.py                      # Współdzielone funkcje (pętle turniejowe, eksport JSON)
├── negamax.py                    # Eksperyment 1: Wpływ głębokości (Depth) na wyniki
├── negamaxAB.py                  # Eksperyment 2: Testowanie odcięcia Alfa-Beta i pomiar czasu
├── expecti-minimax.py            # Eksperyment 3: Expectiminimax vs standardowy Negamax
├── negamaxAB_results.json        # Wygenerowane raporty w formacie JSON do eksperymentu 2.
├── expectiminimax-results.json   # Wygenerowane raporty w formacie JSON do eksperymentu 3.
├── wykres1.png                   # Wygenerowany wykres z eksperymentu 1
├── report.pdf                    # Końcowy raport z analizą wyników
└── requirements.txt              # Zależności projektowe
```

### Uruchamianie eksperymentów

1. Stwórz środowisko wirtualne i zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```

2. Uruchom wybrany eksperyment, aby rozpocząć symulacje turniejowe i wygenerować wyniki:
   ```bash
    python negamax.py
    python negamaxAB.py
    python expecti-minimax.py
   ```
   

## Projekt 3: Podstawy Gymnasium (Q-Learning i Deep Q-Network)

Projekt implementuje i porównuje algorytmy uczenia ze wzmocnieniem w dwóch środowiskach biblioteki **`Gymnasium`**: dyskretnym **CliffWalking** oraz ciągłym **CartPole-v1**.

---

### Środowisko 1: CliffWalking

Klasyczny problem planowania na siatce 4×12, w którym agent musi dotrzeć z lewego do prawego dolnego rogu siatki, omijając krawędź urwiska. Przestrzeń stanów i akcji jest w pełni dyskretna, co pozwala na bezpośrednie zastosowanie Q-Learningu z tablicą wartości.

Przeprowadzono eksperymenty z trzema strategiami wygaszania parametru ε, porównując ich wpływ na szybkość konwergencji i stabilność wyników.

### Środowisko 2: CartPole-v1

Problem sterowania odwróconym wahadłem z **ciągłą** przestrzenią obserwacji (4 zmienne rzeczywiste). Zaimplementowano i porównano dwa algorytmy:

- **Q-Learning z dyskretyzacją** – każda zmienna stanu jest mapowana na kubełki (`np.digitize`); punkt odniesienia pokazujący ograniczenia podejścia tablicowego
- **DQN (Deep Q-Network)** – sieć neuronowa aproksymuje Q(s,a) bezpośrednio na ciągłym wektorze stanu; implementacja oparta na oficjalnym tutorialu PyTorch z soft update target network (τ=0.01)

Przeprowadzono przeszukiwanie hiperparametrów dla dwóch architektur sieci (128×128 i 128×64) z metryką optymalizacji opartą na całkowitej zdyskontowanej nagrodzie.

---

### Struktura plików

```text
gymnasium/
├── cliff_walking/
│   ├── q_learning_agent.py     # Klasa QLearningAgent (tablica Q, strategia ε-greedy)
│   ├── cliff_walking.ipynb     # Eksperymenty: trzy strategie wygaszania ε, krzywe uczenia
│   └── experiments.ipynb       # Dodatkowe eksperymenty
│
└── cart_pole/
    ├── cartpole_agent.py       # Klasa CartPoleQLearningAgent (dyskretyzacja + tablica Q)
    ├── dqn_model.py            # Architektury sieci: DQN_128x128 i DQN_128x64
    ├── dqn_train.py            # Funkcja treningowa DQN (tutorial PyTorch) + metryka discounted_total()
    ├── experiments.ipynb       # Pełny notatnik: Q-learning, przeszukiwanie HP, porównanie końcowe
    ├── cartpole_visualize.py   # Wizualizacja wytrenowanego agenta w pygame (wymaga best_policy_net.pt)
    └── best_policy_net.pt      # Wagi najlepszej sieci (generowane przez experiments.ipynb)
```

---

### Uruchamianie

1. Zainstaluj zależności:
   ```bash
   pip install gymnasium torch numpy matplotlib pygame
   ```

2. **CliffWalking** – otwórz i uruchom notatnik:
   ```bash
   jupyter notebook cliff_walking/cliff_walking.ipynb
   ```

3. **CartPole** – uruchom pełny notatnik (trenuje oba algorytmy i zapisuje model):
   ```bash
   jupyter notebook cart_pole/experiments.ipynb
   ```

4. **Wizualizacja** – po wygenerowaniu `best_policy_net.pt` przez notatnik:
   ```bash
   python cart_pole/cartpole_visualize.py
   ```
   Sterowanie: **R** – reset epizodu, **Q** – wyjście.
