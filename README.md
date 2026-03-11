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
   
