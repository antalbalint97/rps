class CopycatStrategy:
    name = "CopycatStrategy"

    def __init__(self):
        self.last_opponent_move = "rock"

    def play(self):
        return self.last_opponent_move

    def handle_moves(self, own_move, opponent_move):
        self.last_opponent_move = opponent_move