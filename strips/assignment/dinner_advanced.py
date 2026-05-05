# -*- coding: utf-8 -*-
import sys
import os
import time
import multiprocessing
import queue

base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
aipython_dir = os.path.join(base_dir, 'aipython')
sys.path.insert(0, base_dir)
sys.path.insert(0, aipython_dir)

from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips, boolean
from assignment.reachable_states import count_reachable_states

def clean(x):   return f'clean_{x}'
def quiet(x):   return f'quiet_{x}'
def garbage(x): return f'garbage_{x}'
def dinner(x):  return f'dinner_{x}'
def present(x): return f'present_{x}'

def create_domain(places):
    fdd = {}
    for x in places:
        fdd[clean(x)]   = boolean
        fdd[quiet(x)]   = boolean
        fdd[garbage(x)] = boolean
        fdd[dinner(x)]  = boolean
        fdd[present(x)] = boolean
    actions = set()
    for x in places:
        actions.add(Strips(f'cook_{x}',  {clean(x): True},   {dinner(x): True}))
        actions.add(Strips(f'wrap_{x}',  {quiet(x): True},   {present(x): True}))
        actions.add(Strips(f'carry_{x}', {garbage(x): True}, {garbage(x): False, clean(x): False}))
        actions.add(Strips(f'dolly_{x}', {garbage(x): True}, {garbage(x): False, quiet(x): False}))
    return STRIPS_domain(fdd, actions)

def create_state(places, garbage_places=None):
    gp = set(garbage_places or [])
    state = {}
    for x in places:
        state[clean(x)]   = True
        state[quiet(x)]   = True
        state[garbage(x)] = (x in gp)
        state[dinner(x)]  = False
        state[present(x)] = False
    return state

def heuristic(state, goal):
    return sum(1 for k, v in goal.items() if state.get(k) != v)

def _solve_worker(state, goal, domain_places, use_heuristic, result_queue):
    sys.path.insert(0, base_dir)
    sys.path.insert(0, aipython_dir)
    from aipython.searchMPP import SearcherMPP
    from aipython.stripsForwardPlanner import Forward_STRIPS
    dom  = create_domain(domain_places)
    prob = Planning_problem(dom, state, goal)
    hfn  = heuristic if use_heuristic else None
    planner = Forward_STRIPS(prob, hfn) if hfn else Forward_STRIPS(prob)
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

def run_with_timeout(state, goal, domain_places, use_heuristic, timeout=500):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_solve_worker,
        args=(state, goal, domain_places, use_heuristic, q)
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, True
    try:
        result = q.get(timeout=1)
        return result, False
    except queue.Empty:
        return None, False

def _w(msg, f):
    print(msg)
    f.write(msg + '\n')
    f.flush()

def solve_timed(init, goal, dplaces, use_heuristic, label, f, timeout=500):
    _w(f'\n--- {label} ---', f)
    t0 = time.perf_counter()
    result, timed_out = run_with_timeout(init, goal, dplaces, use_heuristic, timeout)
    elapsed = time.perf_counter() - t0
    if timed_out:
        _w(f'Przekroczono limit czasu ({timeout}s).', f)
        return
    if result is None:
        _w('Nie znaleziono planu.', f)
        return
    cost, actions, _ = result
    _w(f'Znaleziono plan. Koszt: {cost}', f)
    for i, a in enumerate(actions, 1):
        _w(f'  {i}. {a}', f)
    _w(f'Czas: {elapsed:.4f}s', f)

def solve_subgoals_timed(subgoals, init, dplaces, use_heuristic, label, f, timeout=500):
    _w(f'\n--- {label} ---', f)
    state = init.copy()
    full_plan = []
    t0 = time.perf_counter()
    for i, sg in enumerate(subgoals, 1):
        result, timed_out = run_with_timeout(state, sg, dplaces, use_heuristic, timeout)
        if timed_out:
            _w(f'Przekroczono limit czasu ({timeout}s) dla podcelu {i}.', f)
            return
        if result is None:
            _w(f'Brak planu dla podcelu {i}.', f)
            return
        cost, actions, state = result
        _w(f'\nPodcel {i}: {sg}\nAkcje: {actions}', f)
        full_plan.extend(actions)
    elapsed = time.perf_counter() - t0
    _w('\nPelny plan z podcelami:', f)
    for i, a in enumerate(full_plan, 1):
        _w(f'  {i}. {a}', f)
    _w(f'Czas: {elapsed:.4f}s', f)

p4_din = list('abcde');  p4_pre = list('fghij');  places4 = p4_din + p4_pre
p5_din = list('abcdef'); p5_pre = list('ghijk');  places5 = p5_din + p5_pre
p6_din = list('abcdef'); p6_pre = list('ghijkl'); places6 = p6_din + p6_pre

def make_goal(din, pre, all_places):
    """garbage=False w celu wymusza użycie carry/dolly dla każdego miejsca."""
    g = {}
    for x in din:
        g[dinner(x)]  = True
    for x in pre:
        g[present(x)] = True
    for x in all_places:
        g[garbage(x)] = False
    return g

goal4 = make_goal(p4_din, p4_pre, places4)
goal5 = make_goal(p5_din, p5_pre, places5)
goal6 = make_goal(p6_din, p6_pre, places6)

subgoals4 = [
    make_goal(p4_din[:2], p4_pre[:2], p4_din[:2] + p4_pre[:2]),
    make_goal(p4_din[:4], p4_pre[:4], p4_din[:4] + p4_pre[:4]),
    goal4,
]
subgoals5 = [
    make_goal(p5_din[:2], p5_pre[:2], p5_din[:2] + p5_pre[:2]),
    make_goal(p5_din[:4], p5_pre[:3], p5_din[:4] + p5_pre[:3]),
    goal5,
]
subgoals6 = [
    make_goal(p6_din[:2], p6_pre[:2], p6_din[:2] + p6_pre[:2]),
    make_goal(p6_din[:4], p6_pre[:4], p6_din[:4] + p6_pre[:4]),
    make_goal(p6_din,     p6_pre[:4], p6_din     + p6_pre[:4]),
    goal6,
]


if __name__ == '__main__':
    multiprocessing.freeze_support()

    OUTPUT_FILE = os.path.join(base_dir, 'results_dinner_advanced.txt')

    problems = [
        ('Problem 4 [8 pkt] - 20 akcji', places4, goal4, subgoals4),
        ('Problem 5 [8 pkt] - 22 akcje', places5, goal5, subgoals5),
        ('Problem 6 [8 pkt] - 24 akcje', places6, goal6, subgoals6),
    ]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        _w('DINNER ADVANCED - PROBLEMY NA 8 PUNKTOW', f)
        _w('=' * 60, f)

        for label, places, goal, subgoals in problems:
            _w(f'\n{"=" * 60}', f)
            _w(label, f)
            _w('=' * 60, f)

            dom  = create_domain(places)
            init = create_state(places, places)

            # n = count_reachable_states(dom, init)
            # _w(f'Osiagalne stany (BFS): {n}', f)

            solve_timed(init, goal, places, False, 'Rozwiazanie bez heurystyki',               f)
            solve_timed(init, goal, places, True,  'Rozwiazanie z heurystyka',                 f)
            solve_subgoals_timed(subgoals, init, places, False, 'Rozwiazanie z podcelami',                  f)
            solve_subgoals_timed(subgoals, init, places, True,  'Rozwiazanie z podcelami i heurystyka',     f)

    print(f'\nWyniki zapisane do: {OUTPUT_FILE}')