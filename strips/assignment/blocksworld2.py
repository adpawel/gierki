# -*- coding: utf-8 -*-
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')

sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from reachable_states import count_reachable_states
from solver import solve, solve_subgoals
from assignment.reachable_states import count_reachable_states
from solver import init_output_file, write_section

blocks = {'A', 'B', 'C', 'D'}
tables = {'T1', 'T2'}

def on(x, y):
    return f'on_{x}_{y}'

def clear(x):
    return f'clear_{x}'

def heuristic(state, goal):
    h = 0

    target_support = {
        'A': 'B',
        'B': 'C',
        'C': 'D',
    }

    for x, y in target_support.items():
        if state.get(on(x, y), False):
            continue

        h += 2

        if not state.get(clear(x), False):
            h += 1

        if y in blocks and not state.get(clear(y), False):
            h += 1

    return h

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

initial_state = {}

# domyślnie wszystko False
for b in blocks:
    initial_state[clear(b)] = False
    for t in tables:
        initial_state[on(b, t)] = False
    for b2 in blocks:
        if b != b2:
            initial_state[on(b, b2)] = False

# stan początkowy:
# A na B, B na C, C na T1, D na T2
initial_state[on('A', 'B')] = True
initial_state[on('B', 'C')] = True
initial_state[on('C', 'T1')] = True
initial_state[on('D', 'T2')] = True

# clear są tylko bloki bez niczego na sobie
initial_state[clear('A')] = True
initial_state[clear('B')] = False
initial_state[clear('C')] = False
initial_state[clear('D')] = True

goal = {
    on('C', 'D'): True,
    on('B', 'C'): True,
    on('A', 'B'): True
}

problem = Planning_problem(domain, initial_state, goal)

subgoals = [
    {on('C', 'D'): True},
    {on('C', 'D'): True, on('B', 'C'): True},
    {on('C', 'D'): True, on('B', 'C'): True, on('A', 'B'): True},
]

OUTPUT_FILE = os.path.join(base_dir, 'results_blocksworld2.txt')

init_output_file(OUTPUT_FILE, 'BLOCKSWORLD - RESULTS')

reachable = count_reachable_states(domain, initial_state)
print('reachable states:', reachable)
with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Osiągalne stany: {reachable}\n")

print('\n--- Rozwiązanie bez heurystyki ---\n')
write_section(OUTPUT_FILE, 'Rozwiązanie bez heurystyki')
solve(problem, output_file=OUTPUT_FILE)

print('\n--- Rozwiązanie z heurystyką ---\n')
write_section(OUTPUT_FILE, 'Rozwiązanie z heurystyką')
solve(problem, heuristic, output_file=OUTPUT_FILE)

print('\n--- Rozwiązanie z podcelami ---\n')
write_section(OUTPUT_FILE, 'Rozwiązanie z podcelami')
solve_subgoals(subgoals, initial_state, domain, output_file=OUTPUT_FILE)

print('\n--- Rozwiązanie z podcelami i heurystyką ---\n')
write_section(OUTPUT_FILE, 'Rozwiązanie z podcelami i heurystyką')
solve_subgoals(subgoals, initial_state, domain, heuristic, output_file=OUTPUT_FILE)

print(f'\nWyniki zapisane do: {OUTPUT_FILE}')