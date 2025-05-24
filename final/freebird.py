from rpsa_sdk.strategy import Strategy
from rpsa_sdk.helpers import counter_move
import random
import math
from collections import defaultdict, Counter, deque

class FreeBird(Strategy):
    name = "FreeBird"

    def __init__(self, model=None):
        super().__init__(model)
        self.n = 2
        self.decay = 0.9
        self.laplace_k = 1.0
        self.confidence_threshold = 0.4

        self.moves = ["rock", "paper", "scissors"]
        self.context_counts = defaultdict(lambda: defaultdict(float))
        self.context_total = defaultdict(float)
        self.history = deque(maxlen=self.n)
        self.mode = "bayesian"

    def _compute_smoothed_probs(self, context):
        total = self.context_total[context]
        move_counts = self.context_counts[context]
        return {
            move: (move_counts[move] + self.laplace_k) / (total + self.laplace_k * len(self.moves))
            for move in self.moves
        }

    def _get_confidence(self, probs):
        return max(probs.values()) - min(probs.values())

    def play(self):
        context = tuple(self.history)
        if context not in self.context_counts or self.context_total[context] == 0:
            return random.choice(self.moves)
        smoothed_probs = self._compute_smoothed_probs(context)
        confidence = self._get_confidence(smoothed_probs)
        if confidence < self.confidence_threshold:
            return random.choice(self.moves)
        expected_rewards = {
            "rock": smoothed_probs["scissors"],
            "paper": smoothed_probs["rock"],
            "scissors": smoothed_probs["paper"]
        }
        return max(expected_rewards, key=expected_rewards.get)

    def handle_moves(self, own_move, opponent_move):
        context = tuple(self.history)
        if context in self.context_counts:
            for move in self.moves:
                self.context_counts[context][move] *= self.decay
            self.context_total[context] *= self.decay
        self.context_counts[context][opponent_move] += 1.0
        self.context_total[context] += 1.0
        self.history.append((own_move, opponent_move))

strategy = FreeBird