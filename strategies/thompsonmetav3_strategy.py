import random
from collections import defaultdict
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class ThompsonMetaV3_Contextual:
    name = "ThompsonMetaV3_Contextual"

    def __init__(self):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }

        # Track performance per (strategy, context)
        self.stats = defaultdict(lambda: {"score_sum": 0.0, "count": 1.0})  # Avoid divide-by-zero

        self.last_used = None
        self.last_move = None
        self.opp_history = []
        self.decay_factor = 0.98

    def get_context(self):
        if len(self.opp_history) < 3:
            return ("_", "_", "_")
        return tuple(self.opp_history[-3:])  # Last 3 opponent moves

    def play(self):
        context = self.get_context()

        # Thompson-like sampling per strategy in this context
        samples = {}
        for name in self.strategies:
            stat = self.stats[(name, context)]
            mean = stat["score_sum"] / stat["count"]
            samples[name] = random.gauss(mean, 1.0)

        # Pick best strategy for this context
        self.last_used = max(samples, key=samples.get)
        self.last_move = self.strategies[self.last_used].play()
        return self.last_move

    def handle_moves(self, my_move, opponent_move):
        self.opp_history.append(opponent_move)

        # Update all sub-strategies
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        # Score: +1 win, -1 loss, 0 tie
        score = self.get_score_delta(my_move, opponent_move)

        # Current context
        context = self.get_context()

        # Apply decay to all buckets in this context
        for name in self.strategies:
            self.stats[(name, context)]["score_sum"] *= self.decay_factor
            self.stats[(name, context)]["count"] *= self.decay_factor

        # Update only the selected strategy
        self.stats[(self.last_used, context)]["score_sum"] += score
        self.stats[(self.last_used, context)]["count"] += 1

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
