# -*- coding: utf-8 -*-
import sys
import os
import time

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')

sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from solver import solve, solve_subgoals
from assignment.reachable_states import count_reachable_states
from solver import init_output_file, write_section

def op1(x1, x2, x3):
    return f'op1_{x1}_{x2}_{x3}'

def op2(x1, x2, x3):
    return f'op2_{x1}_{x2}_{x3}'

def S(x, y):
    return f'S_{x}_{y}'

def R(x, y):
    return f'R_{x}_{y}'

objects = {'a', 'b', 'c', 'd', 'e'}

feature_domain_dict = {}
for x in objects:
    for y in objects:
        feature_domain_dict[S(x, y)] = boolean
        feature_domain_dict[R(x, y)] = boolean

actions = set()

for x1 in objects:
    for x2 in objects:
        for x3 in objects:
            actions.add(
                Strips(
                    op1(x1, x2, x3),
                    {S(x1, x2): True, R(x3, x1): True},
                    {S(x2, x1): True, S(x1, x3): True, R(x3, x1): False}
                )
            )

            actions.add(
                Strips(
                    op2(x1, x2, x3),
                    {S(x3, x1): True, R(x2, x2): True},
                    {S(x1, x3): True, S(x3, x1): False}
                )
            )

domain = STRIPS_domain(feature_domain_dict, actions)

initial_state = {}
for x in objects:
    for y in objects:
        initial_state[S(x, y)] = False
        initial_state[R(x, y)] = False

initial_state[S('a', 'b')] = True
initial_state[S('b', 'c')] = True
initial_state[S('c', 'd')] = True
initial_state[S('d', 'e')] = True

for x in objects:
    initial_state[R(x, x)] = True

initial_state[R('c', 'a')] = True
initial_state[R('d', 'b')] = True
initial_state[R('e', 'c')] = True

goal = {
    S('b', 'a'): True,
    S('c', 'b'): True,
    S('d', 'c'): True,
    S('e', 'd'): True,
}

problem = Planning_problem(domain, initial_state, goal)

subgoal1 = {
    S('b', 'a'): True
}

subgoal2 = {
    S('b', 'a'): True,
    S('c', 'b'): True
}

subgoal3 = {
    S('b', 'a'): True,
    S('c', 'b'): True,
    S('d', 'c'): True
}

subgoals = [subgoal1, subgoal2, subgoal3, goal]

def heuristic(state, goal):
    return sum(1 for feature, value in goal.items() if state.get(feature) != value)

OUTPUT_FILE = os.path.join(base_dir, 'results_problem1.txt')

init_output_file(OUTPUT_FILE, 'PROBLEM 1 - RESULTS')

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