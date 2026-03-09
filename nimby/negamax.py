from easyAI import AI_Player, Negamax
from nim import Nim
import random


def play_match(ai1, ai2, is_probabilistic=False):
    """
    Rozgrywa jedną partię i zwraca ID wygranego gracza (1 lub 2).
    """
    game = Nim([ai1, ai2])
    
    while not game.is_over():
        # AI wymyśla najlepszy ruch (w pełni deterministycznie)
        move = game.player.ask_move(game)
        
        # Jeśli gramy w wariant Nimby, aplikujemy losowość do ZATWIERDZONEGO ruchu
        if is_probabilistic:
            pile, take = map(int, move.split(","))
            # 10% szansy na wzięcie o 1 mniej (jeśli bierze więcej niż 1)
            if random.random() < 0.1 and take > 1:
                take -= 1
                move = f"{pile},{take}"
                
        game.make_move(move)
        game.switch_player()
        
    # Zwycięzcą jest ten, kto wykonał ostatni ruch (czyli gracz poprzedni)
    return 3 - game.current_player


def run_tournament(ai1, ai2, is_probabilistic, num_games=20):
    """
    Gra serię spotkań, zmieniając gracza rozpoczynającego w połowie.
    """
    wins = {1: 0, 2: 0} # 1 to ai1, 2 to ai2
    
    # Pierwsza połowa gier: ai1 zaczyna (jest graczem nr 1 w easyAI)
    for _ in range(num_games // 2):
        winner = play_match(ai1, ai2, is_probabilistic)
        wins[winner] += 1
        
    # Druga połowa gier: ai2 zaczyna (staje się graczem nr 1 w tej konkretnej partii)
    for _ in range(num_games // 2):
        winner = play_match(ai2, ai1, is_probabilistic)
        actual_winner = 2 if winner == 1 else 1
        wins[actual_winner] += 1
        
    return wins

if __name__ == "__main__":
    ai_depth_3 = AI_Player(Negamax(3))
    ai_depth_5 = AI_Player(Negamax(3))
    
    games_total = 100
    
    print(f"Rozpoczynam turniej ({games_total} gier w każdym wariancie)...")
    
    print("\n--- TEST 1: Deterministyczny Nim (Zwykły) ---")
    wins_det = run_tournament(ai_depth_3, ai_depth_5, is_probabilistic=False, num_games=games_total)
    print(f"Negamax(depth=3) wygrane: {wins_det[1]}")
    print(f"Negamax(depth=3) wygrane: {wins_det[2]}")
    
    print("\n--- TEST 2: Probabilistyczny Nimby (10% szansy na błąd) ---")
    wins_prob = run_tournament(ai_depth_3, ai_depth_5, is_probabilistic=True, num_games=games_total)
    print(f"Negamax(depth=3) wygrane: {wins_prob[1]}")
    print(f"Negamax(depth=3) wygrane: {wins_prob[2]}")