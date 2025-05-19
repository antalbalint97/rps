import random
import math
from collections import defaultdict
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class ThompsonMetaV2:
    name = "ThompsonMetaV2"

    def __init__(self):
        # Sub-strategies
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }

        # Performance scores and decay-based tracking
        self.stats = {
            name: {"score_sum": 0.0, "count": 1.0}  # initialize with count=1 to avoid div by zero
            for name in self.strategies
        }

        self.last_used = None
        self.last_move = None
        self.decay_factor = 0.98  # decay applied to all strategy stats every round

    def play(self):
        # Thompson Sampling with Gaussian-like sampling from score mean
        samples = {
            name: random.gauss(stat["score_sum"] / stat["count"], 1.0)
            for name, stat in self.stats.items()
        }

        # Select strategy with highest sampled performance
        self.last_used = max(samples, key=samples.get)
        self.last_move = self.strategies[self.last_used].play()
        return self.last_move

    def handle_moves(self, my_move, opponent_move):
        # Update all sub-strategies with the round outcome
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        # Compute score delta: +1 for win, 0 for tie, -1 for loss
        score = self.get_score_delta(my_move, opponent_move)

        # Apply decay to all stats to reduce weight of old results
        for stat in self.stats.values():
            stat["score_sum"] *= self.decay_factor
            stat["count"] *= self.decay_factor

        # Update selected strategy’s score and count
        self.stats[self.last_used]["score_sum"] += score
        self.stats[self.last_used]["count"] += 1

    def get_score_delta(self, move1, move2):
        if move1 == move2:
            return 0
        elif (
            (move1 == "rock" and move2 == "scissors") or
            (move1 == "scissors" and move2 == "paper") or
            (move1 == "paper" and move2 == "rock")
        ):
            return 1
        else:
            return -1
