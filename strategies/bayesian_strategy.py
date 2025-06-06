import random
import math
from collections import defaultdict, deque

class BayesianNGramStrategy:
    name = "BayesianNGramStrategy_MultiMode"

    def __init__(self, n=2, decay=0.9, laplace_k=1.0, confidence_threshold=0.4):
        self.n = n
        self.decay = decay
        self.laplace_k = laplace_k
        self.confidence_threshold = confidence_threshold

        self.moves = ["rock", "paper", "scissors"]
        self.context_counts = defaultdict(lambda: defaultdict(float))
        self.context_total = defaultdict(float)
        self.history = deque(maxlen=n)

        self.mode = "bayesian"  # or "random"

    def _compute_smoothed_probs(self, context):
        total = self.context_total[context]
        move_counts = self.context_counts[context]

        return {
            move: (move_counts[move] + self.laplace_k) / (total + self.laplace_k * len(self.moves))
            for move in self.moves
        }

    def _calculate_entropy(self, probs):
        return -sum(p * math.log2(p) for p in probs.values() if p > 0)

    def _get_confidence(self, probs):
        # Confidence is high when one move has a clearly higher probability
        return max(probs.values()) - min(probs.values())

    def play(self):
        context = tuple(self.history)

        if context not in self.context_counts or self.context_total[context] == 0:
            self.mode = "random"
            return random.choice(self.moves)

        smoothed_probs = self._compute_smoothed_probs(context)
        confidence = self._get_confidence(smoothed_probs)

        if confidence < self.confidence_threshold:
            self.mode = "random"
            return random.choice(self.moves)
        else:
            self.mode = "bayesian"
            expected_rewards = {
                "rock": smoothed_probs["scissors"],
                "paper": smoothed_probs["rock"],
                "scissors": smoothed_probs["paper"]
            }
            return max(expected_rewards, key=expected_rewards.get)

    def handle_moves(self, own_move, opponent_move):
        context = tuple(self.history)

        # Decay old counts
        if context in self.context_counts:
            for move in self.moves:
                self.context_counts[context][move] *= self.decay
            self.context_total[context] *= self.decay

        # Update counts
        self.context_counts[context][opponent_move] += 1.0
        self.context_total[context] += 1.0

        # Update history
        self.history.append((own_move, opponent_move))
