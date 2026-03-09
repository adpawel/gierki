from easyAI import AI_Player, Negamax
from nim import Nim
import random
import time


class NegamaxNoAB:
    """
    Własna implementacja algorytmu Negamax bez odcięcia alfa-beta.
    """
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, game):
        best_move = None
        best_score = -float('inf')
        
        # Sprawdzamy wszystkie możliwe ruchy na danym poziomie
        for move in game.possible_moves():
            game.make_move(move)
            # Wywołujemy rekurencję z minusem (zasada negamax)
            score = -self._negamax(game, self.depth - 1)
            game.unmake_move(move)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _negamax(self, game, depth):
        # Jeśli osiągnęliśmy koniec przeszukiwania lub koniec gry
        if depth == 0 or game.is_over():
            return game.scoring()
        
        best_score = -float('inf')
        for move in game.possible_moves():
            game.make_move(move)
            score = -self._negamax(game, depth - 1)
            game.unmake_move(move)
            if score > best_score:
                best_score = score
        return best_score


def play_match(ai1, ai2, is_probabilistic=False):
    """Zwraca ID wygranego gracza oraz czasy myślenia (czas_gracza_1, czas_gracza_2)."""
    game = Nim([ai1, ai2])
    times = {1: 0.0, 2: 0.0}
    
    while not game.is_over():
        start_time = time.time()
        move = game.player.ask_move(game)
        end_time = time.time()
        
        # Zapisujemy czas myślenia aktualnego gracza
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


def run_tournament(ai1, ai2, name1, name2, is_probabilistic, num_games=10):
    wins = {1: 0, 2: 0}
    total_times = {1: 0.0, 2: 0.0} # 1 to ai1, 2 to ai2
    
    for _ in range(num_games // 2):
        winner, times = play_match(ai1, ai2, is_probabilistic)
        wins[winner] += 1
        total_times[1] += times[1]
        total_times[2] += times[2]
        
    for _ in range(num_games // 2):
        winner, times = play_match(ai2, ai1, is_probabilistic)
        actual_winner = 2 if winner == 1 else 1
        wins[actual_winner] += 1
        # Tutaj gracz 2 był "graczem 1" w logice silnika gry
        total_times[2] += times[1]
        total_times[1] += times[2]
        
    print(f"Wyniki starcia: {name1} vs {name2} (Gry: {num_games})")
    print(f"[{name1}] Wygrane: {wins[1]} | Łączny czas myślenia: {total_times[1]:.4f} sek")
    print(f"[{name2}] Wygrane: {wins[2]} | Łączny czas myślenia: {total_times[2]:.4f} sek\n")


if __name__ == "__main__":
    # Inicjalizacja graczy z i bez Alfa-Beta dla dwóch głębokości
    ai_ab_4 = AI_Player(Negamax(4))       # Domyślny easyAI (Z Alfa-Beta)
    ai_no_ab_4 = AI_Player(NegamaxNoAB(4)) # Nasz (BEZ Alfa-Beta)
    
    ai_ab_5 = AI_Player(Negamax(5))
    ai_no_ab_5 = AI_Player(NegamaxNoAB(5))

    games = 20 # Mniejsza liczba, bo Negamax bez AB dla depth=5 liczy się długo!

    print("================ DETERMINISTYCZNY NIM ================\n")
    # Porównanie Negamax(z AB) vs Negamax(bez AB) na tej samej głębokości 
    run_tournament(ai_ab_4, ai_no_ab_4, "Negamax(AB, depth=4)", "Negamax(NoAB, depth=4)", False, games)
    run_tournament(ai_ab_5, ai_no_ab_5, "Negamax(AB, depth=5)", "Negamax(NoAB, depth=5)", False, games)
    
    print("================ PROBABILISTYCZNY NIMBY (10%) ================\n")
    # To samo dla wersji probabilistycznej 
    run_tournament(ai_ab_4, ai_no_ab_4, "Negamax(AB, depth=4)", "Negamax(NoAB, depth=4)", True, games)
    run_tournament(ai_ab_5, ai_no_ab_5, "Negamax(AB, depth=5)", "Negamax(NoAB, depth=5)", True, games)

    # Porównanie różnych głębokości (dodatkowo, dla pełności zadania z pdf)
    print("================ RÓŻNE GŁĘBOKOŚCI (Alfa-Beta) ================\n")
    run_tournament(ai_ab_4, ai_ab_5, "Negamax(AB, depth=4)", "Negamax(AB, depth=5)", True, 20)