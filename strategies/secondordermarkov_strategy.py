import random
import math
from collections import defaultdict, Counter

class SecondOrderMarkov:
    name = "SecondOrderMarkov"

    def __init__(self, order=2, alpha=0.2):
        self.order = order
        self.alpha = alpha
        self.history = []
        self.transition_probs = defaultdict(lambda: defaultdict(float))

    def counter_move(self, move):
        if move == "rock":
            return "paper"
        elif move == "paper":
            return "scissors"
        elif move == "scissors":
            return "rock"
        return random.choice(["rock", "paper", "scissors"])

    def play(self):
        if len(self.history) < self.order:
            return random.choice(["rock", "paper", "scissors"])
        key = tuple(self.history[-self.order:])
        next_probs = self.transition_probs.get(key, {})
        if not next_probs:
            return random.choice(["rock", "paper", "scissors"])
        predicted = max(next_probs, key=next_probs.get)
        return self.counter_move(predicted)

    def handle_moves(self, own_move, opponent_move):
        if len(self.history) >= self.order:
            key = tuple(self.history[-self.order:])
            for move in ["rock", "paper", "scissors"]:
                observed = 1.0 if move == opponent_move else 0.0
                old = self.transition_probs[key][move]
                self.transition_probs[key][move] = (1 - self.alpha) * old + self.alpha * observed
        self.history.append(opponent_move)
