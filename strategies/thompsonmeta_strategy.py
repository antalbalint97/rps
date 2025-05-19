import random
from collections import defaultdict
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class ThompsonMetaStrategy:
    name = "ThompsonMetaStrategy"

    def __init__(self):
        # Sub-strategies
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }

        # Stats for Thompson Sampling
        self.stats = {
            "markov": {"wins": 1, "losses": 1},
            "enhanced": {"wins": 1, "losses": 1},
            "qlearn": {"wins": 1, "losses": 1}
        }

        self.last_used = None
        self.last_move = None

    def play(self):
        # Sample from each strategy's Beta distribution
        samples = {
            name: random.gammavariate(s["wins"] + 1, 1) /
                  (random.gammavariate(s["losses"] + 1, 1) + 1e-9)
            for name, s in self.stats.items()
        }

        # Select strategy with highest sample
        self.last_used = max(samples, key=samples.get)
        self.last_move = self.strategies[self.last_used].play()
        return self.last_move

    def handle_moves(self, my_move, opponent_move):
        # Update sub-strategies
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        # Update performance
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
