# -*- coding: utf-8 -*-
import sys
import os
import time

from aipython.stripsProblem import Planning_problem

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'aipython'))

from aipython.searchMPP import SearcherMPP
from aipython.stripsForwardPlanner import Forward_STRIPS


def _print(message, output_file=None):
    """Print to console and optionally to file."""
    print(message)
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


def init_output_file(filename, title):
    """Initialize output file with title.
    
    Args:
        filename: Path to output file
        title: Title to write at the beginning
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write('='*60 + '\n\n')


def write_section(filename, title):
    """Write section header to output file.
    
    Args:
        filename: Path to output file
        title: Section title
    """
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {title} ---\n\n")


def extract_actions(path):
    actions = []
    current = path
    while current.arc is not None:
        actions.append(current.arc.action)
        current = current.initial
    actions.reverse()
    return actions


def print_plan(actions, prefix="Plan", output_file=None):
    _print(f"{prefix}:", output_file)
    for i, action in enumerate(actions, 1):
        _print(f"  {i}. {action}", output_file)


def search(problem, heuristic=None):
    planner = Forward_STRIPS(problem, heuristic) if heuristic else Forward_STRIPS(problem)
    return SearcherMPP(planner).search()


def solve(problem, heuristic=None, output_file=None):
    t0 = time.perf_counter()
    path = search(problem, heuristic)
    elapsed = time.perf_counter() - t0

    if path is None:
        _print("Nie znaleziono planu.", output_file)
        return

    actions = extract_actions(path)
    _print(f"Znaleziono plan. Koszt: {path.cost}", output_file)
    print_plan(actions, output_file=output_file)
    _print(f"Czas: {elapsed:.4f}s", output_file)


def solve_subgoals(subgoals, initial_state, domain, heuristic=None, output_file=None):
    current_state = initial_state.copy()
    full_plan = []
    t0 = time.perf_counter()

    for i, subgoal in enumerate(subgoals, 1):
        path = search(Planning_problem(domain, current_state, subgoal), heuristic)

        if path is None:
            _print(f"Nie znaleziono planu dla podcelu {i}.", output_file)
            break

        actions = extract_actions(path)
        _print(f"\nPodcel {i}: {subgoal}\nAkcje: {actions}", output_file)
        full_plan.extend(actions)

        current_state = path.end().assignment.copy()

    elapsed = time.perf_counter() - t0
    print_plan(full_plan, "\nPełny plan z podcelami", output_file)
    _print(f"Czas: {elapsed:.4f}s", output_file)