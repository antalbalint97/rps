import random


class NoiseInjectionStrategy:
    name = "NoiseInjectionStrategy"

    def __init__(self, entropy=0.4):
        self.entropy = entropy
        self.last_move = None
        self.moves = ["rock", "paper", "scissors"]

    def _counter_move(self, move):
        if move == "rock":
            return "paper"
        elif move == "paper":
            return "scissors"
        elif move == "scissors":
            return "rock"
        return random.choice(self.moves)

    def play(self):
        base_move = self.last_move or random.choice(self.moves)
        if random.random() < self.entropy:
            move = random.choice(self.moves)
        else:
            move = self._counter_move(base_move)
        self.last_move = move
        return move

    def handle_moves(self, own_move, opponent_move):
        pass  # This strategy is stateless
