from collections import deque
from aipython.stripsForwardPlanner import Forward_STRIPS
from aipython.stripsProblem import Planning_problem

#* BFS
def count_reachable_states(domain, initial_state):
    problem = Planning_problem(domain, initial_state, {})
    planner = Forward_STRIPS(problem)
    start = planner.start_node()
    frontier = deque([start])
    visited = {start}

    while frontier:
        state = frontier.popleft()
        for arc in planner.neighbors(state):
            next_state = arc.to_node
            if next_state not in visited:
                visited.add(next_state)
                frontier.append(next_state)
    
    return len(visited)