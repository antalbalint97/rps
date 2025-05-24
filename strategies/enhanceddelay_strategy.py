import random
import math
from collections import defaultdict, Counter


class EnhancedStrategyDelay:
    name = "EnhancedStrategy"

    def __init__(self, delay=0, response_chance=0.8):
        self.history = []
        self.delay = delay
        self.response_chance = response_chance

    def counter_move(self, move):
        if move == "rock":
            return "paper"
        elif move == "paper":
            return "scissors"
        elif move == "scissors":
            return "rock"
        return random.choice(["rock", "paper", "scissors"])

    def play(self):
        if len(self.history) <= self.delay or random.random() > self.response_chance:
            return random.choice(["rock", "paper", "scissors"])
        delayed_index = -1 - self.delay
        move_to_counter = self.history[delayed_index][1]
        return self.counter_move(move_to_counter)

    def handle_moves(self, own_move, opponent_move):
        self.history.append((own_move, opponent_move))
