# -*- coding: utf-8 -*-
import sys
import os
import time
import multiprocessing

from aipython.stripsProblem import Planning_problem

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'aipython'))

TIMEOUT = 40

def _search_worker(initial_state, goal, prob_domain, heuristic_fn, result_queue):
    """Worker uruchamiany w osobnym procesie."""
    sys.path.insert(0, base_dir)
    sys.path.insert(0, os.path.join(base_dir, 'aipython'))
    from aipython.searchMPP import SearcherMPP
    from aipython.stripsForwardPlanner import Forward_STRIPS

    prob = Planning_problem(prob_domain, initial_state, goal)
    if heuristic_fn is not None:
        planner = Forward_STRIPS(prob, heuristic_fn)
    else:
        planner = Forward_STRIPS(prob)
    path = SearcherMPP(planner).search()

    if path is None:
        result_queue.put(None)
        return

    actions = []
    cur = path
    while cur.arc is not None:
        actions.append(cur.arc.action)
        cur = cur.initial
    actions.reverse()
    result_queue.put((path.cost, actions, path.end().assignment.copy()))


def _run_with_timeout(initial_state, goal, prob_domain, heuristic_fn, timeout):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_search_worker,
        args=(initial_state, goal, prob_domain, heuristic_fn, q),
        daemon=True
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, True
    return (q.get() if not q.empty() else None), False


# ── public API ────────────────────────────────────────────────────────────────

def _print(message, output_file=None):
    print(message)
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def init_output_file(filename, title):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write('=' * 60 + '\n\n')

def write_section(filename, title):
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

def solve(problem, heuristic=None, output_file=None, timeout=TIMEOUT):
    t0 = time.perf_counter()
    result, timed_out = _run_with_timeout(
        problem.initial_state, problem.goal, problem.prob_domain,
        heuristic, timeout
    )
    elapsed = time.perf_counter() - t0

    if timed_out:
        _print(f"Przekroczono limit czasu ({timeout}s).", output_file)
        return
    if result is None:
        _print("Nie znaleziono planu.", output_file)
        return

    cost, actions, _ = result
    _print(f"Znaleziono plan. Koszt: {cost}", output_file)
    print_plan(actions, output_file=output_file)
    _print(f"Czas: {elapsed:.4f}s", output_file)

def solve_subgoals(subgoals, initial_state, domain, heuristic=None, output_file=None, timeout=TIMEOUT):
    state = initial_state.copy()
    full_plan = []
    t0 = time.perf_counter()

    for i, subgoal in enumerate(subgoals, 1):
        result, timed_out = _run_with_timeout(
            state, subgoal, domain, heuristic, timeout
        )
        if timed_out:
            _print(f"Przekroczono limit czasu ({timeout}s) dla podcelu {i}.", output_file)
            break
        if result is None:
            _print(f"Nie znaleziono planu dla podcelu {i}.", output_file)
            break
        cost, actions, state = result
        _print(f"\nPodcel {i}: {subgoal}\nAkcje: {actions}", output_file)
        full_plan.extend(actions)

    elapsed = time.perf_counter() - t0
    print_plan(full_plan, "\nPelny plan z podcelami", output_file)
    _print(f"Czas: {elapsed:.4f}s", output_file)