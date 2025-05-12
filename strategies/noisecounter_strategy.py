import random

class NoiseCounterStrategy:
    name = "NoiseCounterStrategy"

    def __init__(self, counter_prob=0.8):
        self.counter_prob = counter_prob
        self.last_opponent_move = None

    def play(self):
        if self.last_opponent_move and random.random() < self.counter_prob:
            return self.counter(self.last_opponent_move)
        else:
            return random.choice(["rock", "paper", "scissors"])

    def handle_moves(self, my_move, opponent_move):
        self.last_opponent_move = opponent_move

    def counter(self, move):
        return {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }[move]
