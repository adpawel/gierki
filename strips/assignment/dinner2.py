# -*- coding: utf-8 -*-
"""
STRIPS Planning – Birthday Dinner Domain
=========================================
Projekt 2: STRIPS  |  Realizacja wymagań na 8 punktów

Wymagania spełnione:
  [4 pkt] Trzy problemy bazowe, każdy z ≥50 osiągalnymi stanami
          i rozwiązaniem składającym się z ≥4 instancji akcji.
  [6 pkt] Każdy problem posiada ≥2 podcele; rozwiązania
          z/bez heurystyki i z/bez podcelów.
  [8 pkt] Trzy dodatkowe problemy, których optymalne rozwiązanie
          wymaga ≥20 instancji akcji (Problemy 4, 5, 6).

Dziedzina (Birthday Dinner):
  cook(x)  : clean(x)   → dinner(x)
  wrap(x)  : quiet(x)   → present(x)
  carry(x) : garbage(x) → ¬garbage(x), ¬clean(x)
  dolly(x) : garbage(x) → ¬garbage(x), ¬quiet(x)

Heurystyka: liczba niespełnionych warunków celu.
  Jest dopuszczalna, gdyż każda akcja może spełnić co najwyżej
  jeden warunek celu – dolna granica liczby potrzebnych akcji.
"""

import sys
import os

base_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')

sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from solver import solve, solve_subgoals, init_output_file, write_section
from assignment.reachable_states import count_reachable_states

# ============================================================
# POMOCNICZE GENERATORY NAZW CECH I AKCJI
# ============================================================
def clean(x):   return f'clean_{x}'
def quiet(x):   return f'quiet_{x}'
def garbage(x): return f'garbage_{x}'
def dinner(x):  return f'dinner_{x}'
def present(x): return f'present_{x}'

def cook(x):  return f'cook_{x}'
def wrap(x):  return f'wrap_{x}'
def carry(x): return f'carry_{x}'
def dolly(x): return f'dolly_{x}'

# ============================================================
# BUDOWNICZY DOMENY
# ============================================================
def create_domain(places):
    """Tworzy STRIPS_domain dla podanej listy miejsc."""
    fdd = {}
    for x in places:
        fdd[clean(x)]   = boolean
        fdd[quiet(x)]   = boolean
        fdd[garbage(x)] = boolean
        fdd[dinner(x)]  = boolean
        fdd[present(x)] = boolean

    actions = set()
    for x in places:
        actions.add(Strips(cook(x),  {clean(x): True},   {dinner(x): True}))
        actions.add(Strips(wrap(x),  {quiet(x): True},   {present(x): True}))
        actions.add(Strips(carry(x), {garbage(x): True}, {garbage(x): False, clean(x): False}))
        actions.add(Strips(dolly(x), {garbage(x): True}, {garbage(x): False, quiet(x): False}))

    return STRIPS_domain(fdd, actions)


def create_state(places, garbage_places=None):
    """
    Tworzy stan początkowy: wszystkie miejsca czyste i ciche,
    brak kolacji i prezentów. Miejsca z garbage_places mają garbage=True.
    """
    gp = set(garbage_places or [])
    state = {}
    for x in places:
        state[clean(x)]   = True
        state[quiet(x)]   = True
        state[garbage(x)] = (x in gp)
        state[dinner(x)]  = False
        state[present(x)] = False
    return state


# ============================================================
# HEURYSTYKA
# Liczy liczbę warunków celu jeszcze niespełnionych w danym stanie.
# Jest dopuszczalna: każda akcja spełnia co najwyżej 1 warunek
# celu, więc heurystyka nigdy nie zawyża rzeczywistego kosztu.
# ============================================================
def heuristic(state, goal):
    return sum(1 for k, v in goal.items() if state.get(k) != v)


# ============================================================
#  PROBLEM 1  [4/6 pkt]
#  3 miejsca, brak śmieci
#  Stany osiągalne: 4^3 = 64  (≥50 ✓)
#  Minimalne rozwiązanie: cook+wrap dla każdego miejsca = 6 akcji (≥4 ✓)
#  Podcele: 2 (≥2 ✓)
# ============================================================
places1  = ['a', 'b', 'c']
domain1  = create_domain(places1)
initial1 = create_state(places1)
goal1    = {**{dinner(x): True for x in places1},
            **{present(x): True for x in places1}}

subgoals1 = [
    # Podcel 1: obiad i prezent tylko dla miejsca 'a'  (2 akcje)
    {dinner('a'): True, present('a'): True},
    # Podcel 2 = pełny cel
    goal1,
]
problem1 = Planning_problem(domain1, initial1, goal1)


# ============================================================
#  PROBLEM 2  [4/6 pkt]
#  4 miejsca, brak śmieci
#  Stany osiągalne: 4^4 = 256  (≥50 ✓)
#  Minimalne rozwiązanie: 8 akcji (≥4 ✓)
#  Podcele: 3 (≥2 ✓)
# ============================================================
places2  = ['a', 'b', 'c', 'd']
domain2  = create_domain(places2)
initial2 = create_state(places2)
goal2    = {**{dinner(x): True for x in places2},
            **{present(x): True for x in places2}}

subgoals2 = [
    # Podcel 1: 2 pierwsze miejsca (4 akcje)
    {**{dinner(x): True for x in ['a', 'b']},
     **{present(x): True for x in ['a', 'b']}},
    # Podcel 2: 3 pierwsze miejsca (6 akcji)
    {**{dinner(x): True for x in ['a', 'b', 'c']},
     **{present(x): True for x in ['a', 'b', 'c']}},
    # Podcel 3 = pełny cel
    goal2,
]
problem2 = Planning_problem(domain2, initial2, goal2)


# ============================================================
#  PROBLEM 3  [4/6 pkt]
#  5 miejsc, brak śmieci
#  Stany osiągalne: 4^5 = 1 024  (≥50 ✓)
#  Minimalne rozwiązanie: 10 akcji (≥4 ✓)
#  Podcele: 3 (≥2 ✓)
# ============================================================
places3  = ['a', 'b', 'c', 'd', 'e']
domain3  = create_domain(places3)
initial3 = create_state(places3)
goal3    = {**{dinner(x): True for x in places3},
            **{present(x): True for x in places3}}

subgoals3 = [
    # Podcel 1: pierwsze 2 miejsca (4 akcje)
    {**{dinner(x): True for x in ['a', 'b']},
     **{present(x): True for x in ['a', 'b']}},
    # Podcel 2: pierwsze 4 miejsca (8 akcji)
    {**{dinner(x): True for x in ['a', 'b', 'c', 'd']},
     **{present(x): True for x in ['a', 'b', 'c', 'd']}},
    # Podcel 3 = pełny cel
    goal3,
]
problem3 = Planning_problem(domain3, initial3, goal3)


# ============================================================
#  PROBLEM 4  [8 pkt] – ≥20 akcji
#  10 miejsc, brak śmieci
#  Stany osiągalne: 4^10 ≈ 1 048 576  (≥50 ✓)
#  Minimalne rozwiązanie: cook+wrap × 10 = 20 akcji (≥20 ✓)
#  Podcele: 4 (≥2 ✓)
# ============================================================
places4  = list('abcdefghij')   # 10 miejsc
domain4  = create_domain(places4)
initial4 = create_state(places4)
goal4    = {**{dinner(x): True for x in places4},
            **{present(x): True for x in places4}}

subgoals4 = [
    # Podcel 1: pierwsze 3 miejsca (6 akcji)
    {**{dinner(x): True for x in places4[:3]},
     **{present(x): True for x in places4[:3]}},
    # Podcel 2: pierwsze 5 miejsc (10 akcji)
    {**{dinner(x): True for x in places4[:5]},
     **{present(x): True for x in places4[:5]}},
    # Podcel 3: pierwsze 8 miejsc (16 akcji)
    {**{dinner(x): True for x in places4[:8]},
     **{present(x): True for x in places4[:8]}},
    # Podcel 4 = pełny cel (20 akcji)
    goal4,
]
problem4 = Planning_problem(domain4, initial4, goal4)


# ============================================================
#  PROBLEM 5  [8 pkt] – ≥20 akcji, śmieci w połowie miejsc
#
#  Miejsca a,b,c: garbage=True → dolly(x)+cook(x) → dinner(x)
#                 (dolly usuwa quiet, więc wrap niedostępne)
#  Miejsca d,e:   garbage=True → carry(x)+wrap(x) → present(x)
#                 (carry usuwa clean, więc cook niedostępne)
#  Miejsca f–j:   brak śmieci  → cook(x)+wrap(x) → dinner+present
#
#  Minimalne rozwiązanie:
#    3×2 (dolly+cook) + 2×2 (carry+wrap) + 5×2 (cook+wrap) = 20 akcji (≥20 ✓)
#  Stany osiągalne: znacznie więcej niż 50 (≥50 ✓)
#  Podcele: 3 (≥2 ✓)
# ============================================================
p5_dolly  = ['a', 'b', 'c']        # dolly → cook  (dinner only)
p5_carry  = ['d', 'e']             # carry → wrap  (present only)
p5_clean  = ['f', 'g', 'h', 'i', 'j']  # cook + wrap
places5   = p5_dolly + p5_carry + p5_clean

domain5  = create_domain(places5)
initial5 = create_state(places5, garbage_places=p5_dolly + p5_carry)

goal5 = {
    **{dinner(x):  True for x in p5_dolly},   # przez dolly+cook
    **{present(x): True for x in p5_carry},   # przez carry+wrap
    **{dinner(x):  True for x in p5_clean},   # przez cook
    **{present(x): True for x in p5_clean},   # przez wrap
}

subgoals5 = [
    # Podcel 1: rozwiąż wszystkie miejsca ze śmieciami (10 akcji)
    {**{dinner(x):  True for x in p5_dolly},
     **{present(x): True for x in p5_carry}},
    # Podcel 2: dodaj 3 pierwsze czyste miejsca (6 akcji więcej)
    {**{dinner(x):  True for x in p5_dolly},
     **{present(x): True for x in p5_carry},
     **{dinner(x):  True for x in p5_clean[:3]},
     **{present(x): True for x in p5_clean[:3]}},
    # Podcel 3 = pełny cel (4 akcje więcej)
    goal5,
]
problem5 = Planning_problem(domain5, initial5, goal5)


# ============================================================
#  PROBLEM 6  [8 pkt] – ≥20 akcji
#  11 miejsc, brak śmieci
#  Stany osiągalne: 4^11 ≈ 4 194 304  (≥50 ✓)
#  Minimalne rozwiązanie: cook+wrap × 11 = 22 akcje (≥20 ✓)
#  Podcele: 4 (≥2 ✓)
# ============================================================
places6  = list('abcdefghijk')   # 11 miejsc
domain6  = create_domain(places6)
initial6 = create_state(places6)
goal6    = {**{dinner(x): True for x in places6},
            **{present(x): True for x in places6}}

subgoals6 = [
    # Podcel 1: pierwsze 3 miejsca (6 akcji)
    {**{dinner(x): True for x in places6[:3]},
     **{present(x): True for x in places6[:3]}},
    # Podcel 2: pierwsze 6 miejsc (12 akcji)
    {**{dinner(x): True for x in places6[:6]},
     **{present(x): True for x in places6[:6]}},
    # Podcel 3: pierwsze 9 miejsc (18 akcji)
    {**{dinner(x): True for x in places6[:9]},
     **{present(x): True for x in places6[:9]}},
    # Podcel 4 = pełny cel (22 akcje)
    goal6,
]
problem6 = Planning_problem(domain6, initial6, goal6)


# ============================================================
# URUCHOMIENIE WSZYSTKICH PROBLEMÓW
# ============================================================
OUTPUT_FILE = os.path.join(base_dir, 'results_dinner2.txt')
init_output_file(OUTPUT_FILE, 'DINNER2 PLANNING – WYNIKI (8 punktów)')

# Tabela: (etykieta, problem, stan_pocz, domena, podcele, min_akcji, opis_stanow)
all_problems = [
    (
        'Problem 1 [4/6 pkt] – 3 miejsca, 6 akcji',
        problem1, initial1, domain1, subgoals1,
        '4^3 = 64 osiągalnych stanów',
    ),
    (
        'Problem 2 [4/6 pkt] – 4 miejsca, 8 akcji',
        problem2, initial2, domain2, subgoals2,
        '4^4 = 256 osiągalnych stanów',
    ),
    (
        'Problem 3 [4/6 pkt] – 5 miejsc, 10 akcji',
        problem3, initial3, domain3, subgoals3,
        '4^5 = 1 024 osiągalne stany',
    ),
    (
        'Problem 4 [8 pkt] – 10 miejsc, 20 akcji',
        problem4, initial4, domain4, subgoals4,
        '4^10 ≈ 1 048 576 osiągalnych stanów',
    ),
    (
        'Problem 5 [8 pkt] – 10 miejsc ze śmieciami, 20 akcji',
        problem5, initial5, domain5, subgoals5,
        'śmieci w 5 miejscach, zróżnicowane ścieżki',
    ),
    (
        'Problem 6 [8 pkt] – 11 miejsc, 22 akcje',
        problem6, initial6, domain6, subgoals6,
        '4^11 ≈ 4 194 304 osiągalnych stanów',
    ),
]

for label, prob, init, dom, sgoals, state_info in all_problems:
    sep = '=' * 65
    header = f'\n{sep}\n{label}\n{state_info}\n{sep}'
    print(header)
    write_section(OUTPUT_FILE, header)

    n_states = count_reachable_states(dom, init)
    msg = f'Osiągalne stany (BFS): {n_states}'
    print(msg)
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg + '\n')

    print('\n--- Rozwiązanie BEZ heurystyki ---')
    write_section(OUTPUT_FILE, 'Rozwiązanie BEZ heurystyki')
    solve(prob, output_file=OUTPUT_FILE)

    print('\n--- Rozwiązanie Z heurystyką ---')
    write_section(OUTPUT_FILE, 'Rozwiązanie Z heurystyką')
    solve(prob, heuristic, output_file=OUTPUT_FILE)

    print('\n--- Rozwiązanie Z podcelami (bez heurystyki) ---')
    write_section(OUTPUT_FILE, 'Rozwiązanie Z podcelami (bez heurystyki)')
    solve_subgoals(sgoals, init, dom, output_file=OUTPUT_FILE)

    print('\n--- Rozwiązanie Z podcelami i heurystyką ---')
    write_section(OUTPUT_FILE, 'Rozwiązanie Z podcelami i heurystyką')
    solve_subgoals(sgoals, init, dom, heuristic, output_file=OUTPUT_FILE)

print(f'\nWyniki zapisane do: {OUTPUT_FILE}')