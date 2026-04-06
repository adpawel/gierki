# -*- coding: utf-8 -*-
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')

sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from solver import solve, solve_subgoals
from assignment.reachable_states import count_reachable_states

# (define (domain birthday-dinner)
#  (:requirements :strips)
#  (:action cook
#     :parameters (?x)
#     :precondition (and (clean ?x))
#     :effect (and (dinner ?x)))
#  (:action wrap
#     :parameters (?x)
#     :precondition (and (quiet ?x))
#     :effect (and (present ?x)))
#  (:action carry
#     :parameters (?x)
#     :precondition (and (garbage ?x))
#     :effect (not (garbage ?x) not (clean ?x)))
#  (:action dolly
#     :parameters (?x)
#     :precondition (and (garbage ?x))
#     :effect (not (garbage ?x) not (quiet ?x)))
# )

#* ACTIONS
def cook(x):
    return f'cook_{x}'

def wrap(x):
    return f'wrap_{x}'

def carry(x):
    return f'carry_{x}'

def dolly(x):
    return f'dolly_{x}'

#* WLASNOSCI
def clean(x):
    return f'clean_{x}'

def quiet(x):
    return f'quiet_{x}'

def garbage(x):
    return f'garbage_{x}'

def dinner(x):
    return f'dinner_{x}'

def present(x):
    return f'present_{x}'

#* DOMAIN
places = {'a', 'b', 'c', 'd', 'e'}  # kitchen, living room etc.

feature_domain_dict = {}
for x in places:
    feature_domain_dict[clean(x)] = boolean
    feature_domain_dict[quiet(x)] = boolean
    feature_domain_dict[garbage(x)] = boolean
    feature_domain_dict[dinner(x)] = boolean
    feature_domain_dict[present(x)] = boolean

#* STRIPS
actions = set()

for x in places:
    actions.add(Strips(cook(x), {clean(x): True}, {dinner(x): True}))
    actions.add(Strips(wrap(x), {quiet(x): True}, {present(x): True}))
    actions.add(Strips(carry(x), {garbage(x): True}, {garbage(x): False, clean(x): False}))
    actions.add(Strips(dolly(x), {garbage(x): True}, {garbage(x): False, quiet(x): False}))

domain = STRIPS_domain(feature_domain_dict, actions)

#* PROBLEM

initial_state = {
    clean('a'): True,
    quiet('a'): True,
    garbage('a'): False,
    dinner('a'): False,
    present('a'): False,

    clean('b'): True,
    quiet('b'): True,
    garbage('b'): False,
    dinner('b'): False,
    present('b'): False,

    clean('c'): True,
    quiet('c'): True,
    garbage('c'): False,
    dinner('c'): False,
    present('c'): False,

    clean('d'): True,
    quiet('d'): True,
    garbage('d'): False,
    dinner('d'): False,
    present('d'): False,

    clean('e'): True,
    quiet('e'): True,
    garbage('e'): False,
    dinner('e'): False,
    present('e'): False,
}

goal = {
    dinner('a'): True,
    present('a'): True,
    dinner('b'): True,
    present('b'): True,
    dinner('c'): True,
    present('c'): True,
    dinner('d'): True,
    present('d'): True,
    dinner('e'): True,
    present('e'): True,
}

problem = Planning_problem(domain, initial_state, goal)

#? ile warunków celu nie jest jeszcze spełnionych
def heuristic(state, goal):
    return sum(1 for k, v in goal.items() if state.get(k) != v)

#* SUBGOALS

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

subgoals = [
    {dinner('a'): True, present('a'): True},
    {dinner('a'): True, present('a'): True,
     dinner('b'): True, present('b'): True},
     goal
]

from assignment.reachable_states import count_reachable_states
from solver import init_output_file, write_section

OUTPUT_FILE = os.path.join(base_dir, 'results_dinner.txt')

init_output_file(OUTPUT_FILE, 'DINNER PLANNING - RESULTS')

print('reachable states:', count_reachable_states(domain, initial_state))
with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Osiągalne stany: {count_reachable_states(domain, initial_state)}\n")

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