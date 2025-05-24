import random
import math
from collections import defaultdict, Counter
from strategies.secondordermarkov_strategy import SecondOrderMarkov
from strategies.qlearningv2_strategy import QLearningStrategyV2
from strategies.enhanceddelay_strategy import EnhancedStrategyDelay


class ShinyDiamond:
    name = "ShinyDiamond_MetaMeta"

    def __init__(self, model=None):

        self.strategies = {
            "markov": SecondOrderMarkov(),
            "enhanced": EnhancedStrategyDelay(),
            "qlearn": QLearningStrategyV2()
        }
        self.actions = list(self.strategies.keys())

        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.last_sa = None

        self.opp_history = []
        self.move_counts = Counter()
        self.strategy_scores = defaultdict(int)
        self.strategy_use = defaultdict(int)
        self.meta_state = None

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

        self.false_pattern_sequence = ["rock", "rock"]
        self.false_pattern_index = 0
        self.false_pattern_triggered = False
        self.false_pattern_usable = True
        self.loss_streak = 0
        self.pre_injection_losses = 0
        self.pre_injection_wins = 0
        self.post_injection_wins = 0

        self.opponent_profiles = defaultdict(lambda: Counter())
        self.strategy_streaks = defaultdict(int)
        self.current_streak_strategy = None
        self.classified_opponents = {}

    def play(self):
        state = self.get_profiled_state()

        # Handle false pattern injection — just play and exit early
        if self.false_pattern_triggered:
            move = self.false_pattern_sequence[self.false_pattern_index % len(self.false_pattern_sequence)]
            self.false_pattern_index += 1
            if self.false_pattern_index >= len(self.false_pattern_sequence):
                self.false_pattern_triggered = False
                if self.post_injection_wins <= self.pre_injection_wins:
                    self.false_pattern_usable = False
            return move  # ✅ return move directly, skip strategy logic

        action = self.select_action(state)

        # Skip saving 'injection' as strategy
        if action not in self.strategies:
            return random.choice(["rock", "paper", "scissors"])

        # Track streaks
        if action == self.current_streak_strategy:
            self.strategy_streaks[action] += 1
        else:
            if self.current_streak_strategy:
                self.strategy_streaks[self.current_streak_strategy] = 0
            self.current_streak_strategy = action
            self.strategy_streaks[action] = 1

        # ✅ Save last strategy only if it's real
        self.last_sa = (state, action)
        self.meta_state = action

        return self.strategies[action].play()

    def select_action(self, state):
        if self.loss_streak >= 3 and not self.false_pattern_triggered and self.false_pattern_usable:
            if random.random() < 0.3:
                self.false_pattern_triggered = True
                self.false_pattern_index = 0
                self.pre_injection_wins = 0
                self.post_injection_wins = 0
                # Do not return anything, let `play()` handle injection
                # Just set up the trigger and fall through to normal selection

        opponent_type = self.classify_opponent()
        if opponent_type == "repeater" and random.random() < 0.6:
            return "enhanced"
        elif opponent_type == "switcher" and random.random() < 0.6:
            return "markov"
        elif opponent_type == "adaptive" and random.random() < 0.6:
            return "qlearn"

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        scores = self.q_table[state].copy()
        for action in scores:
            streak_penalty = self.strategy_streaks[action] * 0.1
            scores[action] -= streak_penalty

        return max(scores, key=scores.get)

    def classify_opponent(self):
        if len(self.opp_history) < 20:
            return "unknown"
        repeat_ratio = sum(self.opp_history[i] == self.opp_history[i-1] for i in range(1, len(self.opp_history))) / len(self.opp_history)
        switch_ratio = sum(self.opp_history[i] != self.opp_history[i-1] for i in range(1, len(self.opp_history))) / len(self.opp_history)

        if repeat_ratio > 0.6:
            return "repeater"
        elif switch_ratio > 0.6:
            return "switcher"
        else:
            return "adaptive"

    def opponent_entropy(self):
        count = Counter(self.opp_history[-20:])
        total = sum(count.values())
        if total == 0:
            return 2.0
        return -sum((freq / total) * math.log2(freq / total) for freq in count.values())

    def handle_moves(self, my_move, opponent_move):
    # Update all sub-strategies
        for strat in self.strategies.values():
            strat.handle_moves(my_move, opponent_move)

        # Track opponent history
        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        # Update profiles
        if self.meta_state is not None:
            self.opponent_profiles[self.meta_state][opponent_move] += 1

        # Evaluate score
        score = self.get_score_delta(my_move, opponent_move)

        # Track injection performance
        if self.false_pattern_triggered:
            if score == 1:
                self.post_injection_wins += 1
        else:
            if score == 1:
                self.pre_injection_wins += 1

        # Update loss streak
        if score == -1:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        # Safeguarded Q-table update
        if self.last_sa and self.last_sa[1] in self.actions:
            state, action = self.last_sa
            new_state = self.get_profiled_state()

            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in self.actions}
            if new_state not in self.q_table:
                self.q_table[new_state] = {a: 0.0 for a in self.actions}

            future = max(self.q_table[new_state].values())
            old_value = self.q_table[state][action]

            self.q_table[state][action] += self.learning_rate * (
                score + self.discount_factor * future - old_value
            )

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)

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
        recent_repeat = int(len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2])
        total = sum(self.move_counts.values())
        bias_level = 0
        switch_rate = 0

        if total:
            bias_ratio = max(self.move_counts.values()) / total
            if bias_ratio > 0.6:
                bias_level = 2
            elif bias_ratio > 0.4:
                bias_level = 1
            switch_rate = sum(self.opp_history[i] != self.opp_history[i - 1] for i in range(1, len(self.opp_history))) / total

        switch_bucket = 2 if switch_rate > 0.6 else 1 if switch_rate > 0.3 else 0
        return (recent_repeat, bias_level, switch_bucket)
