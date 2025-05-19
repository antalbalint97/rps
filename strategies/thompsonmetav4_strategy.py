import random
import math
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class ThompsonMetaV4_Profiled:
    name = "ThompsonMetaV4_Profiled"

    def __init__(self):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }

        self.stats = {
            name: {"wins": 1, "losses": 1}  # Beta prior
            for name in self.strategies
        }

        self.last_used = None
        self.last_move = None

        self.opp_history = []
        self.move_transitions = 0
        self.move_repeats = 0
        self.move_counts = Counter()

    def play(self):
        # Profile opponent
        profile = self.get_opponent_profile()

        # Filter: down-weight obviously bad matches
        strategy_weights = self.get_strategy_weights(profile)

        # Thompson Sampling with weights
        samples = {}
        for name, weight in strategy_weights.items():
            s = self.stats[name]
            score = random.gammavariate(s["wins"] + 1, 1) / (random.gammavariate(s["losses"] + 1, 1) + 1e-9)
            samples[name] = score * weight  # apply fit weighting

        self.last_used = max(samples, key=samples.get)
        self.last_move = self.strategies[self.last_used].play()
        return self.last_move

    def handle_moves(self, my_move, opponent_move):
        # Update all sub-strategies
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        # Update opponent behavior stats
        if self.opp_history:
            if opponent_move == self.opp_history[-1]:
                self.move_repeats += 1
            else:
                self.move_transitions += 1
        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        # Get result
        result = self.get_result(my_move, opponent_move)
        if result == 1:
            self.stats[self.last_used]["wins"] += 1
        elif result == -1:
            self.stats[self.last_used]["losses"] += 1

    def get_result(self, move1, move2):
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

    def get_opponent_profile(self):
        total = sum(self.move_counts.values())

        # Prevent division by zero
        if total == 0:
            move_bias = {move: 1/3 for move in ["rock", "paper", "scissors"]}
        else:
            move_bias = {
                move: self.move_counts[move] / total
                for move in ["rock", "paper", "scissors"]
            }

        repeat_rate = self.move_repeats / total if total > 1 else 0.0
        switch_rate = self.move_transitions / total if total > 1 else 0.0

        return {
            "repeat_rate": repeat_rate,
            "switch_rate": switch_rate,
            "move_bias": move_bias
        }

    def get_strategy_weights(self, profile):
        weights = {}
        r = profile["repeat_rate"]
        s = profile["switch_rate"]
        bias = max(profile["move_bias"].values())

        # Rules (tuneable)
        weights["markov"] = 1.0 if s > 0.3 else 0.5  # needs some transition to predict
        weights["enhanced"] = 1.0 if r > 0.4 or bias > 0.5 else 0.6
        weights["qlearn"] = 1.0  # generalist learner — always viable

        return weights
