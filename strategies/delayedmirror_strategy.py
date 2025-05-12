import random
from collections import deque

class DelayedMirrorStrategy:
    name = "DelayedMirrorStrategy"

    def __init__(self, delay=2):
        self.delay = delay
        self.opponent_history = deque(maxlen=100)

    def play(self):
        if len(self.opponent_history) >= self.delay:
            return self.opponent_history[-self.delay]
        else:
            return random.choice(["rock", "paper", "scissors"])

    def handle_moves(self, my_move, opponent_move):
        self.opponent_history.append(opponent_move)
