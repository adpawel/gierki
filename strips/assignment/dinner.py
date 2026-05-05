# -*- coding: utf-8 -*-
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')

sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from solver import solve, solve_subgoals, init_output_file, write_section
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

# * ACTIONS
def cook(x):
    return f'cook_{x}'

def wrap(x):
    return f'wrap_{x}'

def carry(x):
    return f'carry_{x}'

def dolly(x):
    return f'dolly_{x}'

# * WLASNOSCI
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


def create_domain(places):
    feature_domain_dict = {}
    actions = set()

    for x in places:
        feature_domain_dict[clean(x)] = boolean
        feature_domain_dict[quiet(x)] = boolean
        feature_domain_dict[garbage(x)] = boolean
        feature_domain_dict[dinner(x)] = boolean
        feature_domain_dict[present(x)] = boolean

        actions.add(Strips(cook(x), {clean(x): True}, {dinner(x): True}))
        actions.add(Strips(wrap(x), {quiet(x): True}, {present(x): True}))
        actions.add(Strips(carry(x), {garbage(x): True}, {garbage(x): False, clean(x): False}))
        actions.add(Strips(dolly(x), {garbage(x): True}, {garbage(x): False, quiet(x): False}))

    return STRIPS_domain(feature_domain_dict, actions)


def create_initial_state(places):
    state = {}
    for x in places:
        state[clean(x)] = True
        state[quiet(x)] = True
        state[garbage(x)] = False
        state[dinner(x)] = False
        state[present(x)] = False
    return state


def create_goal(places):
    goal = {}
    for x in places:
        goal[dinner(x)] = True
        goal[present(x)] = True
    return goal


def create_subgoals(places):
    p = list(places)

    if len(p) == 3:
        return [
            {dinner(p[0]): True, present(p[0]): True},
            create_goal(places)
        ]

    if len(p) == 4:
        return [
            {
                dinner(p[0]): True, present(p[0]): True,
                dinner(p[1]): True, present(p[1]): True
            },
            {
                dinner(p[0]): True, present(p[0]): True,
                dinner(p[1]): True, present(p[1]): True,
                dinner(p[2]): True, present(p[2]): True
            },
            create_goal(places)
        ]

    return [
        {dinner(p[0]): True, present(p[0]): True},
        {
            dinner(p[0]): True, present(p[0]): True,
            dinner(p[1]): True, present(p[1]): True
        },
        create_goal(places)
    ]


# ? ile warunków celu nie jest jeszcze spełnionych
def heuristic(state, goal):
    return sum(1 for k, v in goal.items() if state.get(k) != v)


def run_problem(label, places, output_file):
    domain = create_domain(places)
    initial_state = create_initial_state(places)
    goal = create_goal(places)
    subgoals = create_subgoals(places)
    problem = Planning_problem(domain, initial_state, goal)

    print(f'\n=== {label} ===')
    write_section(output_file, label)

    reachable = count_reachable_states(domain, initial_state)
    print('reachable states:', reachable)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Osiągalne stany: {reachable}\n")

    print('\n--- Rozwiązanie bez heurystyki ---\n')
    write_section(output_file, 'Rozwiązanie bez heurystyki')
    solve(problem, output_file=output_file)

    print('\n--- Rozwiązanie z heurystyką ---\n')
    write_section(output_file, 'Rozwiązanie z heurystyką')
    solve(problem, heuristic, output_file=output_file)

    print('\n--- Rozwiązanie z podcelami ---\n')
    write_section(output_file, 'Rozwiązanie z podcelami')
    solve_subgoals(subgoals, initial_state, domain, output_file=output_file)

    print('\n--- Rozwiązanie z podcelami i heurystyką ---\n')
    write_section(output_file, 'Rozwiązanie z podcelami i heurystyką')
    solve_subgoals(subgoals, initial_state, domain, heuristic, output_file=output_file)


def main():
    output_file = os.path.join(base_dir, 'results_dinner.txt')
    init_output_file(output_file, 'DINNER PLANNING - RESULTS')

    problems = [
        ('Problem 1 - 3 miejsca', ['a', 'b', 'c']),
        ('Problem 2 - 4 miejsca', ['a', 'b', 'c', 'd']),
        ('Problem 3 - 5 miejsc', ['a', 'b', 'c', 'd', 'e']),
    ]

    for label, places in problems:
        run_problem(label, places, output_file)

    print(f'\nWyniki zapisane do: {output_file}')


if __name__ == '__main__':
    main()