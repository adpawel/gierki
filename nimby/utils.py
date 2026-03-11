import time
import random
from nim import Nim


def play_match(ai1, ai2, is_probabilistic=False):
    """Returns the winning player's ID and thinking times (player_time_1, player_time_2)."""
    game = Nim([ai1, ai2])
    times = {1: 0.0, 2: 0.0}
    
    while not game.is_over():
        start_time = time.time()
        move = game.player.ask_move(game)
        end_time = time.time()
        
        times[game.current_player] += (end_time - start_time)
        
        if is_probabilistic:
            pile, take = map(int, move.split(","))
            if random.random() < 0.1 and take > 1:
                take -= 1
                move = f"{pile},{take}"
                
        game.make_move(move)
        game.switch_player()
        
    winner = 3 - game.current_player
    return winner, times

def run_tournament(ai1, ai2, name1, name2, is_probabilistic, num_games=20):
    wins = {1: 0, 2: 0}
    total_times = {1: 0.0, 2: 0.0}
    
    for _ in range(num_games // 2):
        winner, times = play_match(ai1, ai2, is_probabilistic)
        wins[winner] += 1
        total_times[1] += times[1]
        total_times[2] += times[2]
        
    for _ in range(num_games // 2):
        winner, times = play_match(ai2, ai1, is_probabilistic)
        actual_winner = 2 if winner == 1 else 1
        wins[actual_winner] += 1
        total_times[2] += times[1]
        total_times[1] += times[2]
        
    print(f"Wyniki rozgrywki: {name1} vs {name2} (Gry: {num_games})")
    print(f"[{name1}] Wygrane: {wins[1]} | Czas: {total_times[1]:.4f} sek")
    print(f"[{name2}] Wygrane: {wins[2]} | Czas: {total_times[2]:.4f} sek\n")
    return wins, total_times

def add_to_report(report_list, env_name, ai1, ai2, name1, name2, is_prob, games):
    """
    It starts the tournament, collects the results, formats them as a dictionary (using JSON), 
    and then adds them to the submitted report list.
    """
    wins, times = run_tournament(ai1, ai2, name1, name2, is_prob, games)
    
    record = {
        "environment": env_name,
        "matchup": f"{name1} vs {name2}",
        "games_played": games,
        "results": {
            "player1": {
                "algorithm": name1,
                "wins": wins[1],
                "time_seconds": round(times[1], 4)
            },
            "player2": {
                "algorithm": name2,
                "wins": wins[2],
                "time_seconds": round(times[2], 4)
            }
        }
    }
    report_list.append(record)