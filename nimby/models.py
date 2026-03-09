class NegamaxNoAB:
    """Własna implementacja algorytmu Negamax bez odcięcia alfa-beta."""
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, game):
        best_move = None
        best_score = -float('inf')
        
        for move in game.possible_moves():
            game.make_move(move)
            score = -self._negamax(game, self.depth - 1)
            game.unmake_move(move)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _negamax(self, game, depth):
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


class ExpectiNegamax:
    """Niestandardowy algorytm Expectiminimax dostosowany do struktury Negamax."""
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

        # 1. Symulacja normalnego ruchu
        game.make_move(move)
        score_normal = -self._negamax(game, depth, -beta, -alpha)
        game.unmake_move(move)

        # 2. Symulacja ruchu pechowego
        if take > 1:
            move_slip = f"{pile},{take-1}"
            game.make_move(move_slip)
            score_slip = -self._negamax(game, depth, -beta, -alpha)
            game.unmake_move(move_slip)
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
                break
        return best_score