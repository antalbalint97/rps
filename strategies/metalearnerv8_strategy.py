import random
import math
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class MetaLearnerV8_HybridDeceptiveQ:
    name = "MetaLearnerV8_HybridDeceptiveQ"

    def __init__(self):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }
        self.actions = list(self.strategies.keys())
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.ts_stats = {a: {"wins": 1, "losses": 1} for a in self.actions}

        self.last_used = None
        self.last_state = None

        self.opp_history = []
        self.my_history = []
        self.move_counts = Counter()
        self.move_repeats = 0
        self.move_transitions = 0

        self.strategy_usage = Counter()

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

        self.false_pattern_mode = False
        self.false_pattern_rounds = 0
        self.false_pattern_move = None

        self.score_window = []
        self.window_size = 10

    def play(self):
        state = self.get_profiled_state()

        if len(self.score_window) >= self.window_size:
            recent_sum = sum(self.score_window[-self.window_size:])
            if recent_sum < -2:
                self.epsilon = min(0.5, self.epsilon + 0.05)
                state = (0, 0, 2, 1)

        if not self.false_pattern_mode and random.random() < 0.05:
            self.false_pattern_mode = True
            self.false_pattern_rounds = random.randint(3, 6)
            self.false_pattern_move = random.choice(["rock", "paper", "scissors"])

        if self.false_pattern_mode:
            self.false_pattern_rounds -= 1
            if self.false_pattern_rounds <= 0:
                self.false_pattern_mode = False
            return self.false_pattern_move

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            ts_sampled = {
                a: random.gammavariate(self.ts_stats[a]["wins"], 1) /
                   (random.gammavariate(self.ts_stats[a]["losses"], 1) + 1e-9)
                for a in self.actions
            }
            q_scores = self.q_table[state]
            combined = {
                a: 0.7 * q_scores[a] + 0.3 * ts_sampled[a]
                for a in self.actions
            }
            action = max(combined, key=combined.get)

        self.last_used = action
        self.last_state = state
        self.strategy_usage[action] += 1

        move = self.strategies[action].play()
        self.my_history.append(move)
        return move

    def handle_moves(self, my_move, opponent_move):
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        if len(self.my_history) >= 2 and self.my_history[-2] == opponent_move:
            if random.random() < 0.3:
                self.false_pattern_mode = True
                self.false_pattern_rounds = random.randint(2, 4)
                self.false_pattern_move = random.choice(["rock", "paper", "scissors"])

        if self.opp_history:
            if opponent_move == self.opp_history[-1]:
                self.move_repeats += 1
            else:
                self.move_transitions += 1

        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        reward = self.get_score_delta(my_move, opponent_move)
        self.score_window.append(reward)
        if len(self.score_window) > 50:
            self.score_window.pop(0)

        # Update Q-table and Thompson stats only if state/action tracking is valid
        if self.last_state is not None and self.last_used is not None:
            new_state = self.get_profiled_state()
            future = max(self.q_table[new_state].values())
            old_value = self.q_table[self.last_state][self.last_used]

            self.q_table[self.last_state][self.last_used] += self.learning_rate * (
                reward + self.discount_factor * future - old_value
            )

            if reward == 1:
                self.ts_stats[self.last_used]["wins"] += 1
            elif reward == -1:
                self.ts_stats[self.last_used]["losses"] += 1

    def get_score_delta(self, move1, move2):
        if move1 == move2:
            return 0
        elif (move1 == "rock" and move2 == "scissors") or \
             (move1 == "scissors" and move2 == "paper") or \
             (move1 == "paper" and move2 == "rock"):
            return 1
        else:
            return -1

    def get_profiled_state(self):
        recent_repeat = 1 if len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2] else 0
        total = sum(self.move_counts.values())
        switch_rate = self.move_transitions / total if total > 1 else 0.0

        if total == 0:
            bias_level = 0
        else:
            bias_ratio = max(self.move_counts.values()) / total
            bias_level = 2 if bias_ratio > 0.6 else 1 if bias_ratio > 0.4 else 0

        if switch_rate > 0.6:
            switch_bucket = 2
        elif switch_rate > 0.3:
            switch_bucket = 1
        else:
            switch_bucket = 0

        recent_loss = 1 if sum(self.score_window[-5:]) < -2 else 0

        return (recent_repeat, bias_level, switch_bucket, recent_loss)
