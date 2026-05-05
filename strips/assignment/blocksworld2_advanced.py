# -*- coding: utf-8 -*-
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')

sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from assignment.reachable_states import count_reachable_states
from solver import solve, solve_subgoals, init_output_file, write_section

# =========================
# PARAMETRY
# =========================
blocks = {'A', 'B', 'C', 'D', 'E', 'F'}
tables = {'T1', 'T2'}

# =========================
# FUNKCJE
# =========================
def on(x, y):
    return f'on_{x}_{y}'

def clear(x):
    return f'clear_{x}'

def heuristic(state, goal):
    return sum(1 for f, v in goal.items() if state.get(f, False) != v)

# =========================
# FEATURE DOMAIN
# =========================
feature_domain_dict = {}

for b in blocks:
    feature_domain_dict[clear(b)] = boolean
    for t in tables:
        feature_domain_dict[on(b, t)] = boolean
    for b2 in blocks:
        if b != b2:
            feature_domain_dict[on(b, b2)] = boolean

# =========================
# AKCJE
# =========================
actions = set()

# move table-table
for b in blocks:
    for t1 in tables:
        for t2 in tables:
            if t1 != t2:
                actions.add(Strips(
                    f'move_{b}_{t1}_{t2}',
                    {on(b, t1): True, clear(b): True},
                    {on(b, t2): True, on(b, t1): False}
                ))

# stack
for a in blocks:
    for b in blocks:
        if a != b:
            for t in tables:
                actions.add(Strips(
                    f'stack_{a}_{b}_{t}',
                    {clear(a): True, clear(b): True, on(a, t): True},
                    {on(a, b): True, on(a, t): False, clear(b): False}
                ))

# unstack
for a in blocks:
    for b in blocks:
        if a != b:
            for t in tables:
                actions.add(Strips(
                    f'unstack_{a}_{b}_{t}',
                    {on(a, b): True, clear(a): True},
                    {on(a, t): True, on(a, b): False, clear(b): True}
                ))

domain = STRIPS_domain(feature_domain_dict, actions)

# =========================
# HELPERY
# =========================
def make_empty_state():
    state = {}
    for b in blocks:
        state[clear(b)] = False
        for t in tables:
            state[on(b, t)] = False
        for b2 in blocks:
            if b != b2:
                state[on(b, b2)] = False
    return state


def state_from_stacks(stacks):
    state = make_empty_state()

    for b in blocks:
        state[clear(b)] = True

    for table, stack in stacks.items():
        if not stack:
            continue

        state[on(stack[0], table)] = True

        for lower, upper in zip(stack, stack[1:]):
            state[on(upper, lower)] = True
            state[clear(lower)] = False

        state[clear(stack[-1])] = True

    return state


def goal_from_stack_strict(stack, table):
    """
    Definiuje cel ściśle:
    - bloki w stosie muszą być ułożone we wskazanej kolejności na danym stole
    - żaden blok spoza stosu NIE może leżeć bezpośrednio na tym stole
    Dzięki temu solver nie może znaleźć 'skrótu' pomijającego część klocków.
    """
    goal = {}

    # warunki True: wymagana kolejność w stosie
    goal[on(stack[0], table)] = True
    for lower, upper in zip(stack, stack[1:]):
        goal[on(upper, lower)] = True

    # warunki False: bloki spoza stosu nie mogą być na tym stole
    for b in blocks:
        if b not in stack:
            goal[on(b, table)] = False

    return goal


# =========================
# PROBLEMY (>=20 akcji)
# =========================

def build_problem_4():
    """
    Start:  T1: [A, C, E]  T2: [B, D, F]
    Cel:    T1: [F, D, B, E, C, A]  (odwrócona kolejność obu wież)
    """
    initial_state = state_from_stacks({
        'T1': ['A', 'C', 'E'],
        'T2': ['B', 'D', 'F'],
    })

    target = ['F', 'D', 'B', 'E', 'C', 'A']
    goal = goal_from_stack_strict(target, 'T1')

    subgoals = [
        goal_from_stack_strict(['F', 'D'], 'T1'),
        goal_from_stack_strict(['F', 'D', 'B', 'E'], 'T1'),
        goal,
    ]

    return initial_state, goal, subgoals


def build_problem_5():
    """
    Start:  T1: [F, E, D]  T2: [C, B, A]
    Cel:    T2: [D, A, E, B, F, C]  (przemieszana kolejność)
    """
    initial_state = state_from_stacks({
        'T1': ['F', 'E', 'D'],
        'T2': ['C', 'B', 'A'],
    })

    target = ['D', 'A', 'E', 'B', 'F', 'C']
    goal = goal_from_stack_strict(target, 'T2')

    subgoals = [
        goal_from_stack_strict(['D', 'A'], 'T2'),
        goal_from_stack_strict(['D', 'A', 'E', 'B'], 'T2'),
        goal,
    ]

    return initial_state, goal, subgoals


def build_problem_6():
    """
    Start:  T1: [B, D, F]  T2: [A, C, E]
    Cel:    T1: [E, C, A, F, D, B]  (przeplatane bloki z obu wież)
    """
    initial_state = state_from_stacks({
        'T1': ['B', 'D', 'F'],
        'T2': ['A', 'C', 'E'],
    })

    target = ['E', 'C', 'A', 'F', 'D', 'B']
    goal = goal_from_stack_strict(target, 'T1')

    subgoals = [
        goal_from_stack_strict(['E', 'C'], 'T1'),
        goal_from_stack_strict(['E', 'C', 'A', 'F'], 'T1'),
        goal,
    ]

    return initial_state, goal, subgoals


# =========================
# URUCHOMIENIE
# =========================

problems = [
    ("Problem 4", build_problem_4()),
    ("Problem 5", build_problem_5()),
    ("Problem 6", build_problem_6()),
]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_FILE = os.path.join(base_dir, 'results_blocksworld_advanced.txt')
init_output_file(OUTPUT_FILE, 'BLOCKSWORLD - ADVANCED')

for name, (initial_state, goal, subgoals) in problems:
    write_section(OUTPUT_FILE, name)
    print(f"\n{name}")
    print("=" * 60)

    problem = Planning_problem(domain, initial_state, goal)

    print('\n--- Rozwiązanie bez heurystyki ---\n')
    write_section(OUTPUT_FILE, f'{name} - Rozwiązanie bez heurystyki')
    solve(problem, output_file=OUTPUT_FILE)

    print('\n--- Rozwiązanie z heurystyką ---\n')
    write_section(OUTPUT_FILE, f'{name} - Rozwiązanie z heurystyką')
    solve(problem, heuristic, output_file=OUTPUT_FILE)

    print('\n--- Rozwiązanie z podcelami ---\n')
    write_section(OUTPUT_FILE, f'{name} - Rozwiązanie z podcelami')
    solve_subgoals(subgoals, initial_state, domain, output_file=OUTPUT_FILE)

    print('\n--- Rozwiązanie z podcelami i heurystyką ---\n')
    write_section(OUTPUT_FILE, f'{name} - Rozwiązanie z podcelami i heurystyką')
    solve_subgoals(subgoals, initial_state, domain, heuristic, output_file=OUTPUT_FILE)

print(f'\nWyniki zapisane do: {OUTPUT_FILE}')