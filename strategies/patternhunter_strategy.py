from collections import Counter
import random

class PatternHunterStrategy:
    name = "PatternHunterStrategy"

    def __init__(self):
        self.history = []

    def play(self):
        if len(self.history) < 4:
            return random.choice(["rock", "paper", "scissors"])

        # Build frequency of 2-length patterns
        patterns = [tuple(self.history[i:i+2]) for i in range(len(self.history)-1)]
        counter = Counter(patterns)
        most_common = counter.most_common(1)[0][0][1]  # predict next move
        return self.counter(most_common)

    def handle_moves(self, my_move, opponent_move):
        self.history.append(opponent_move)
        if len(self.history) > 100:
            self.history.pop(0)

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
