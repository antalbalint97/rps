import random
import math
from collections import defaultdict
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy


class BoltzmannMetaStrategy:
    name = "ThompsonMetaV4_Boltzmann"

    def __init__(self, temperature=0.5):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }

        self.stats = {
            name: {"score_sum": 0.0, "count": 1.0}  # avoid divide-by-zero
            for name in self.strategies
        }

        self.last_used = None
        self.last_move = None
        self.decay_factor = 0.98
        self.temperature = temperature

    def play(self):
        # Compute mean scores
        means = {
            name: stat["score_sum"] / stat["count"]
            for name, stat in self.stats.items()
        }

        # Compute softmax probabilities
        exp_vals = {
            name: math.exp(score / self.temperature)
            for name, score in means.items()
        }
        total = sum(exp_vals.values())
        probs = {
            name: val / total
            for name, val in exp_vals.items()
        }

        # Choose strategy probabilistically
        self.last_used = random.choices(list(probs.keys()), weights=probs.values())[0]
        self.last_move = self.strategies[self.last_used].play()
        return self.last_move

    def handle_moves(self, my_move, opponent_move):
        # Update all sub-strategies
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        score = self.get_score_delta(my_move, opponent_move)

        # Apply decay to all stats
        for stat in self.stats.values():
            stat["score_sum"] *= self.decay_factor
            stat["count"] *= self.decay_factor

        # Update selected strategy
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
