import random
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class MetaLearnerV6_AdaptiveQ:
    name = "MetaLearnerV6_AdaptiveQ"

    def __init__(self):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }
        self.actions = list(self.strategies.keys())
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})

        self.last_used = None
        self.last_state = None

        # Profiling
        self.opp_history = []
        self.move_counts = Counter()
        self.move_repeats = 0
        self.move_transitions = 0

        # Track how often each strategy is used
        self.strategy_usage = Counter()

        # Q-learning config
        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def play(self):
        state = self.get_profiled_state()

        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_used = action
        self.last_state = state

        # Track selected strategy
        self.strategy_usage[action] += 1

        return self.strategies[action].play()

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

        # Get reward
        reward = self.get_score_delta(my_move, opponent_move)

        # Update Q-table
        new_state = self.get_profiled_state()
        future = max(self.q_table[new_state].values())
        old_value = self.q_table[self.last_state][self.last_used]

        self.q_table[self.last_state][self.last_used] += self.learning_rate * (
            reward + self.discount_factor * future - old_value
        )

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

    def get_profiled_state(self):
        # Repeat flag
        recent_repeat = 0
        if len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2]:
            recent_repeat = 1

        # Transition rate
        total = sum(self.move_counts.values())
        switch_rate = self.move_transitions / total if total > 1 else 0.0

        # Bias level
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

        # Discretized switch behavior: 0 (static), 1 (some), 2 (frequent)
        if switch_rate > 0.6:
            switch_bucket = 2
        elif switch_rate > 0.3:
            switch_bucket = 1
        else:
            switch_bucket = 0

        return (recent_repeat, bias_level, switch_bucket)
