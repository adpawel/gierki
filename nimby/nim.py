from easyAI import TwoPlayerGame


class Nim(TwoPlayerGame):
    """
    Deterministyczna baza gry Nim.
    Gracz, który zabierze ostatni element, wygrywa.
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
        # Jeśli is_over() to prawda, znaczy to, że poprzedni gracz 
        # zabrał ostatni element. Obecny gracz przegrywa (-100).
        return -100 if self.is_over() else 0
        
    def show(self):
        # Zostawiamy puste, żeby konsola nie została zalana tekstem podczas 100 gier
        pass