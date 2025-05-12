import random
from collections import defaultdict

class MarkovStrategy:
    name = "MarkovStrategy"

    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_move = None

    def play(self):
        if self.last_move is None:
            return random.choice(["rock", "paper", "scissors"])
        next_probs = self.transition_counts[self.last_move]
        if not next_probs:
            return random.choice(["rock", "paper", "scissors"])
        predicted = max(next_probs, key=next_probs.get)
        return self.counter(predicted)

    def handle_moves(self, my_move, opponent_move):
        if self.last_move is not None:
            self.transition_counts[self.last_move][opponent_move] += 1
        self.last_move = opponent_move

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
