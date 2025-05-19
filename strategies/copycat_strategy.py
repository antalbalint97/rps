from collections import deque
import random

class CopycatStrategy:
    name = "CopycatStrategy"

    def __init__(self, memory_size=3):
        self.opponent_history = deque(maxlen=memory_size)

    def play(self):
        if len(self.opponent_history) == 0:
            return random.choice(["rock", "paper", "scissors"])
        # Predict the next move by repeating the last seen
        return self.opponent_history[-1]

    def handle_moves(self, own_move, opponent_move):
        self.opponent_history.append(opponent_move)
