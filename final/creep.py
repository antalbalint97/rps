from rpsa_sdk.strategy import Strategy
from rpsa_sdk.helpers import counter_move
import random
from collections import defaultdict, deque

class Creep(Strategy):
    name = "Reflex"

    def __init__(self, model=None):
        super().__init__(model)
        self.max_n = 3
        self.decay = 0.9
        self.laplace_k = 1.0
        self.confidence_threshold = 0.4
        self.noise_injection_rate = 0.05
        self.reset_threshold = 0.35
        self.strategy_check_interval = 100

        self.history = deque(maxlen=self.max_n)
        self.moves = ["rock", "paper", "scissors"]

        self.context_counts = defaultdict(lambda: defaultdict(float))
        self.context_total = defaultdict(float)

        self.mirror_count = 0
        self.anti_mirror_count = 0
        self.total_moves = 0
        self.win_count = 0
        self.round_count = 0
        self.last_strategy = "pattern"

    def _get_best_context_probs(self):
        recent = list(self.history)
        best_conf = -1
        best_probs = None
        for i in range(1, min(len(recent), self.max_n) + 1):
            context = tuple(recent[-i:])
            total = self.context_total[context]
            if total == 0:
                continue
            move_counts = self.context_counts[context]
            probs = {
                move: (move_counts[move] + self.laplace_k) / (total + self.laplace_k * 3)
                for move in self.moves
            }
            conf = max(probs.values()) - min(probs.values())
            if conf > best_conf:
                best_conf = conf
                best_probs = probs
        return best_probs, best_conf

    def _should_inject_noise(self):
        return random.random() < self.noise_injection_rate

    def _should_reset_memory(self):
        return self.round_count > 0 and (self.win_count / self.round_count) < self.reset_threshold

    def _classify_opponent(self):
        if self.total_moves < 50:
            return "pattern"
        mirror_ratio = self.mirror_count / self.total_moves
        anti_mirror_ratio = self.anti_mirror_count / self.total_moves
        if mirror_ratio > 0.5:
            return "mirror"
        elif anti_mirror_ratio > 0.5:
            return "antimirror"
        return "pattern"

    def _play_pattern(self):
        probs, confidence = self._get_best_context_probs()
        if probs is None or confidence < self.confidence_threshold:
            return random.choice(self.moves)
        expected_rewards = {
            "rock": probs["scissors"],
            "paper": probs["rock"],
            "scissors": probs["paper"]
        }
        return max(expected_rewards, key=expected_rewards.get)

    def play(self):
        if self._should_inject_noise():
            return random.choice(self.moves)

        if self.round_count % self.strategy_check_interval == 0:
            self.last_strategy = self._classify_opponent()

        if self.last_strategy == "mirror":
            return counter_move(self.history[-1][0]) if self.history else random.choice(self.moves)
        elif self.last_strategy == "antimirror":
            return self.history[-1][0] if self.history else random.choice(self.moves)
        else:
            return self._play_pattern()

    def handle_moves(self, own_move, opponent_move):
        self.round_count += 1
        if counter_move(opponent_move) == own_move:
            self.win_count += 1

        if self.history:
            last_own, _ = self.history[-1]
            if opponent_move == last_own:
                self.mirror_count += 1
            elif opponent_move == counter_move(last_own):
                self.anti_mirror_count += 1
        self.total_moves += 1

        recent = list(self.history)
        for i in range(1, min(len(recent), self.max_n) + 1):
            context = tuple(recent[-i:])
            counts = self.context_counts[context]
            for move in self.moves:
                counts[move] *= self.decay
            self.context_total[context] *= self.decay
            counts[opponent_move] += 1.0
            self.context_total[context] += 1.0

        self.history.append((own_move, opponent_move))

        if self._should_reset_memory():
            self.context_counts.clear()
            self.context_total.clear()
            self.win_count = 0
            self.round_count = 0

strategy = Creep
