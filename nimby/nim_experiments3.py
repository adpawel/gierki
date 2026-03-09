from easyAI import TwoPlayerGame, AI_Player, Negamax
import random
import time

class Nim(TwoPlayerGame):
    def __init__(self, players, piles=(3, 4, 5)):
        self.players = players
        self.piles = list(piles)
        self.current_player = 1

    def possible_moves(self):
        moves = []
        for i in range(len(self.piles)):
            for j in range(1, self.piles[i] + 1):
                moves.append(f"{i},{j}")
        return moves

    def make_move(self, move):
        pile, take = map(int, move.split(","))
        self.piles[pile] -= take

    def unmake_move(self, move):
        pile, take = map(int, move.split(","))
        self.piles[pile] += take

    def is_over(self):
        return sum(self.piles) == 0

    def scoring(self):
        return -100 if self.is_over() else 0
        
    def show(self):
        pass


class ExpectiNegamax:
    """
    Niestandardowy algorytm Expectiminimax dostosowany do struktury Negamax,
    zawierający odcięcie Alfa-Beta.
    """
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, game):
        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in game.possible_moves():
            score = self._expected_value(game, move, self.depth - 1, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        return best_move

    def _expected_value(self, game, move, depth, alpha, beta):
        pile, take = map(int, move.split(","))

        # 1. Symulacja normalnego ruchu (Prawdopodobieństwo 90%)
        game.make_move(move)
        score_normal = -self._negamax(game, depth, -beta, -alpha)
        game.unmake_move(move)

        # 2. Symulacja ruchu pechowego (Prawdopodobieństwo 10%)
        if take > 1:
            move_slip = f"{pile},{take-1}"
            game.make_move(move_slip)
            score_slip = -self._negamax(game, depth, -beta, -alpha)
            game.unmake_move(move_slip)
            
            # Wartość oczekiwana węzła "szansy"
            return 0.9 * score_normal + 0.1 * score_slip
        else:
            return score_normal

    def _negamax(self, game, depth, alpha, beta):
        if depth == 0 or game.is_over():
            return game.scoring()

        best_score = -float('inf')
        for move in game.possible_moves():
            score = self._expected_value(game, move, depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
            
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break # Klasyczne odcięcie Alfa-Beta
        return best_score


def play_match(ai1, ai2, is_probabilistic=False):
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
        
    return 3 - game.current_player, times

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
        
    print(f"Wyniki starcia: {name1} vs {name2} (Gry: {num_games})")
    print(f"[{name1}] Wygrane: {wins[1]} | Czas: {total_times[1]:.4f} sek")
    print(f"[{name2}] Wygrane: {wins[2]} | Czas: {total_times[2]:.4f} sek\n")


if __name__ == "__main__":
    ai_negamax_4 = AI_Player(Negamax(4))
    ai_expecti_4 = AI_Player(ExpectiNegamax(4))
    
    ai_negamax_5 = AI_Player(Negamax(5))
    ai_expecti_5 = AI_Player(ExpectiNegamax(5))

    games = 100

    print("================ STARCIĘ TYTANÓW: NEGAMAX vs EXPECTIMINIMAX ================\n")
    print("Testowane na PRAWDZIWEJ, probabilistycznej grze (Nimby 10%)...")
    
    run_tournament(ai_negamax_4, ai_expecti_4, "Zwykły Negamax(4)", "ExpectiMinimax(4)", is_probabilistic=True, num_games=games)
    run_tournament(ai_negamax_5, ai_expecti_5, "Zwykły Negamax(5)", "ExpectiMinimax(5)", is_probabilistic=True, num_games=games)