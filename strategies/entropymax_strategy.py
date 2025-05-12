import random
from collections import deque
import math

class EntropyMaximizerStrategy:
    name = "EntropyMaximizerStrategy"

    def __init__(self, window_size=10, entropy_threshold=1.3):
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.opponent_history = deque(maxlen=window_size)

    def play(self):
        if len(self.opponent_history) < self.window_size:
            return random.choice(["rock", "paper", "scissors"])

        entropy = self.calculate_entropy()
        if entropy < self.entropy_threshold:
            # Opponent is predictable → counter most frequent move
            return self.counter(self.most_common_move())
        else:
            return random.choice(["rock", "paper", "scissors"])

    def handle_moves(self, my_move, opponent_move):
        self.opponent_history.append(opponent_move)

    def most_common_move(self):
        counts = {"rock": 0, "paper": 0, "scissors": 0}
        for move in self.opponent_history:
            counts[move] += 1
        return max(counts, key=counts.get)

    def counter(self, move):
        return {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }[move]

    def calculate_entropy(self):
        counts = {"rock": 0, "paper": 0, "scissors": 0}
        for move in self.opponent_history:
            counts[move] += 1
        total = len(self.opponent_history)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
