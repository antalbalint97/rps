class CycleStrategy:
    name = "CycleStrategy"

    def __init__(self):
        self.moves = ["rock", "paper", "scissors"]
        self.index = 0

    def play(self):
        move = self.moves[self.index]
        self.index = (self.index + 1) % 3
        return move

    def handle_moves(self, own_move, opponent_move):
        pass