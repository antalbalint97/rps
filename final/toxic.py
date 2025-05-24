from rpsa_sdk.strategy import Strategy
from rpsa_sdk.helpers import counter_move
import random
import math
from collections import defaultdict, deque, Counter

class Toxic(Strategy):
    name = "Toxic"

    def __init__(self, model=None):
        super().__init__(model)
        self.max_n = 3
        self.decay = 0.9
        self.laplace_k = 1.0
        self.confidence_threshold = 0.4
        self.noise_injection_rate = 0.05
        self.reset_threshold = 0.35  # win rate threshold to trigger memory reset
        self.history = deque(maxlen=self.max_n)
        self.moves = ["rock", "paper", "scissors"]

        # memory for all contexts
        self.context_counts = defaultdict(lambda: defaultdict(float))
        self.context_total = defaultdict(float)

        # opponent behavior stats
        self.mirror_count = 0
        self.anti_mirror_count = 0
        self.total_moves = 0
        self.win_count = 0
        self.round_count = 0

    def _compute_smoothed_probs(self, context):
        total = self.context_total[context]
        move_counts = self.context_counts[context]
        return {
            move: (move_counts[move] + self.laplace_k) / (total + self.laplace_k * len(self.moves))
            for move in self.moves
        }

    def _get_confidence(self, probs):
        return max(probs.values()) - min(probs.values())

    def _get_best_context_probs(self):
        best_conf = -1
        best_probs = None
        for i in range(1, self.max_n + 1):
            context = tuple(list(self.history)[-i:])
            if self.context_total[context] == 0:
                continue
            probs = self._compute_smoothed_probs(context)
            conf = self._get_confidence(probs)
            if conf > best_conf:
                best_conf = conf
                best_probs = probs
        return best_probs, best_conf

    def _should_inject_noise(self):
        return random.random() < self.noise_injection_rate

    def _should_reset_memory(self):
        if self.round_count == 0:
            return False
        return (self.win_count / self.round_count) < self.reset_threshold

    def play(self):
        if self._should_inject_noise():
            return random.choice(self.moves)

        probs, confidence = self._get_best_context_probs()
        if probs is None or confidence < self.confidence_threshold:
            return random.choice(self.moves)

        expected_rewards = {
            "rock": probs["scissors"],
            "paper": probs["rock"],
            "scissors": probs["paper"]
        }
        return max(expected_rewards, key=expected_rewards.get)

    def handle_moves(self, own_move, opponent_move):
        self.round_count += 1
        if counter_move(opponent_move) == own_move:
            self.win_count += 1

        # opponent classification
        if len(self.history) > 0:
            last_own, last_opp = self.history[-1]
            if opponent_move == last_own:
                self.mirror_count += 1
            elif opponent_move == counter_move(last_own):
                self.anti_mirror_count += 1
        self.total_moves += 1

        # memory update
        for i in range(1, self.max_n + 1):
            context = tuple(list(self.history)[-i:])
            if context:
                for move in self.moves:
                    self.context_counts[context][move] *= self.decay
                self.context_total[context] *= self.decay
                self.context_counts[context][opponent_move] += 1.0
                self.context_total[context] += 1.0

        self.history.append((own_move, opponent_move))

        if self._should_reset_memory():
            self.context_counts.clear()
            self.context_total.clear()
            self.win_count = 0
            self.round_count = 0

strategy = Toxic
