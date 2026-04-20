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

blocks = {'A', 'B', 'C', 'D'}
tables = {'T1', 'T2'}

def on(x, y):
    return f'on_{x}_{y}'

def clear(x):
    return f'clear_{x}'

def heuristic(state, goal):
    return sum(1 for feature, value in goal.items() if state.get(feature, False) != value)

feature_domain_dict = {}

for b in blocks:
    feature_domain_dict[clear(b)] = boolean
    for t in tables:
        feature_domain_dict[on(b, t)] = boolean
    for b2 in blocks:
        if b != b2:
            feature_domain_dict[on(b, b2)] = boolean

actions = set()

# move między stołami
for b in blocks:
    for t1 in tables:
        for t2 in tables:
            if t1 != t2:
                actions.add(Strips(
                    f'move_{b}_{t1}_{t2}',
                    {
                        on(b, t1): True,
                        clear(b): True
                    },
                    {
                        on(b, t2): True,
                        on(b, t1): False
                    }
                ))

# stack z table na block
for a in blocks:
    for b in blocks:
        if a != b:
            for t in tables:
                actions.add(Strips(
                    f'stack_{a}_{b}_{t}',
                    {
                        clear(a): True,
                        clear(b): True,
                        on(a, t): True
                    },
                    {
                        on(a, b): True,
                        on(a, t): False,
                        clear(b): False
                    }
                ))

# unstack z block na table
for a in blocks:
    for b in blocks:
        if a != b:
            for t in tables:
                actions.add(Strips(
                    f'unstack_{a}_{b}_{t}',
                    {
                        on(a, b): True,
                        clear(a): True
                    },
                    {
                        on(a, t): True,
                        on(a, b): False,
                        clear(b): True
                    }
                ))

domain = STRIPS_domain(feature_domain_dict, actions)

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

def build_problem_1():
    state = make_empty_state()

    # A na B, B na C, C na T1, D na T2
    state[on('A', 'B')] = True
    state[on('B', 'C')] = True
    state[on('C', 'T1')] = True
    state[on('D', 'T2')] = True

    state[clear('A')] = True
    state[clear('B')] = False
    state[clear('C')] = False
    state[clear('D')] = True

    goal = {
        on('C', 'D'): True,
        on('B', 'C'): True,
        on('A', 'B'): True
    }

    subgoals = [
        {on('C', 'D'): True},
        {on('C', 'D'): True, on('B', 'C'): True},
        goal
    ]

    return state, goal, subgoals

def build_problem_2():
    state = make_empty_state()

    # A na T1, B na T2, C na D, D na T1
    state[on('A', 'T1')] = True
    state[on('B', 'T2')] = True
    state[on('C', 'D')] = True
    state[on('D', 'T1')] = True

    state[clear('A')] = True
    state[clear('B')] = True
    state[clear('C')] = True
    state[clear('D')] = False

    goal = {
        on('A', 'B'): True,
        on('B', 'C'): True,
        on('C', 'D'): True
    }

    subgoals = [
        {on('C', 'D'): True},
        {on('B', 'C'): True, on('C', 'D'): True},
        goal
    ]

    return state, goal, subgoals

def build_problem_3():
    state = make_empty_state()

    # A na T1, B na A, C na T2, D na T1
    state[on('A', 'T1')] = True
    state[on('B', 'A')] = True
    state[on('C', 'T2')] = True
    state[on('D', 'T1')] = True

    state[clear('A')] = False
    state[clear('B')] = True
    state[clear('C')] = True
    state[clear('D')] = True

    goal = {
        on('D', 'C'): True,
        on('C', 'B'): True,
        on('B', 'A'): True
    }

    subgoals = [
        {on('D', 'C'): True},
        {on('D', 'C'): True, on('C', 'B'): True},
        goal
    ]

    return state, goal, subgoals

problems = [
    ("Problem 1", build_problem_1()),
    ("Problem 2", build_problem_2()),
    ("Problem 3", build_problem_3()),
]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_FILE = os.path.join(base_dir, 'results_blocksworld_all.txt')
init_output_file(OUTPUT_FILE, 'BLOCKSWORLD - ALL PROBLEMS')

for name, (initial_state, goal, subgoals) in problems:
    write_section(OUTPUT_FILE, name)
    print(f"\n{name}")
    print("=" * 60)

    reachable = count_reachable_states(domain, initial_state)
    print('reachable states:', reachable)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Osiągalne stany: {reachable}\n")

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