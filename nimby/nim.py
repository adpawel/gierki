from easyAI import TwoPlayerGame


class Nim(TwoPlayerGame):
    """
    Deterministic basis of the game Nim.
    The player who takes the last piece wins.
    """
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
        # If is_over() is true, it means the previous player took the last item. The current player loses (-100)
        return -100 if self.is_over() else 0
        
    def show(self):
        pass