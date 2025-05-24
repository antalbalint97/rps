import random
import math
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy


class DreamWeaverV7:
    name = "DreamWeaverV7_EntropyBait"

    def __init__(self):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }
        self.actions = list(self.strategies.keys())

        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.last_sa = None

        self.opp_history = []
        self.move_counts = Counter()
        self.move_repeats = 0
        self.move_transitions = 0

        self.strategy_history = []
        self.pattern_threshold = 8
        self.pattern_window = 10

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def play(self) -> str:
        state = self.get_profiled_state()
        action = self.select_action(state)

        self.last_sa = (state, action)
        self.strategy_history.append(action)

        return self.strategies[action].play()

    def select_action(self, state) -> str:
        # Break out of overused strategy patterns
        if len(self.strategy_history) >= self.pattern_window:
            recent = self.strategy_history[-self.pattern_window:]
            most_common = Counter(recent).most_common(1)[0]
            if most_common[1] >= self.pattern_threshold:
                alt_actions = [a for a in self.actions if a != most_common[0]]
                return random.choice(alt_actions)

        # Exploit low-entropy opponent
        if self.opponent_entropy() < 1.2 and random.random() < 0.5:
            return "enhanced"

        # Bait high-entropy opponent
        if self.opponent_entropy() > 1.6 and random.random() < 0.3:
            return random.choice(["markov", "qlearn"])

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation
        return max(self.q_table[state], key=self.q_table[state].get)

    def opponent_entropy(self) -> float:
        count = Counter(self.opp_history[-20:])
        total = sum(count.values())
        if total == 0:
            return 2.0
        return -sum((p := freq / total) * math.log2(p) for freq in count.values())

    def handle_moves(self, my_move: str, opponent_move: str):
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        if self.opp_history and opponent_move == self.opp_history[-1]:
            self.move_repeats += 1
        elif self.opp_history:
            self.move_transitions += 1

        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        # Q-learning meta-update
        if self.last_sa is not None:
            state, action = self.last_sa
            reward = self.get_score_delta(my_move, opponent_move)
            new_state = self.get_profiled_state()
            future = max(self.q_table[new_state].values())
            old_value = self.q_table[state][action]

            self.q_table[state][action] += self.learning_rate * (
                reward + self.discount_factor * future - old_value
            )

        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_score_delta(self, move1: str, move2: str) -> int:
        if move1 == move2:
            return 0
        elif (move1 == "rock" and move2 == "scissors") or \
             (move1 == "scissors" and move2 == "paper") or \
             (move1 == "paper" and move2 == "rock"):
            return 1
        else:
            return -1

    def get_profiled_state(self) -> tuple:
        recent_repeat = int(
            len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2]
        )

        total = sum(self.move_counts.values())
        switch_rate = self.move_transitions / total if total > 1 else 0.0

        if total == 0:
            bias_level = 0
        else:
            bias_ratio = max(self.move_counts.values()) / total
            if bias_ratio > 0.6:
                bias_level = 2
            elif bias_ratio > 0.4:
                bias_level = 1
            else:
                bias_level = 0

        if switch_rate > 0.6:
            switch_bucket = 2
        elif switch_rate > 0.3:
            switch_bucket = 1
        else:
            switch_bucket = 0

        return (recent_repeat, bias_level, switch_bucket)
