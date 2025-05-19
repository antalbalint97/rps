import random
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class AntiMetaV2_MetaPredictor:
    name = "AntiMetaV2_MetaPredictor"

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

        # Opponent profiling
        self.opp_history = []
        self.move_counts = Counter()
        self.move_repeats = 0
        self.move_transitions = 0

        # Inferred learning type (0 = unknown, 1 = reactive, 2 = pattern-seeker, 3 = stochastic)
        self.predicted_type = 0

        # Q-learning config
        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

        # Track strategy usage
        self.strategy_usage = Counter()

    def play(self):
        state = self.get_profiled_state()

        # Adjust Q-values based on predicted opponent type
        if self.predicted_type == 2:  # Pattern seeker → avoid being too predictable
            adjusted = {a: v - 0.1 * i for i, (a, v) in enumerate(sorted(self.q_table[state].items()))}
        elif self.predicted_type == 1:  # Reactive learner → prefer misleading patterns
            adjusted = self.q_table[state].copy()
            adjusted["markov"] += 0.1  # Push Markov if opponent is reacting
        else:
            adjusted = self.q_table[state]

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(adjusted, key=adjusted.get)

        self.last_used = action
        self.last_state = state
        self.strategy_usage[action] += 1

        return self.strategies[action].play()

    def handle_moves(self, my_move, opponent_move):
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        # Update profile
        if self.opp_history:
            if opponent_move == self.opp_history[-1]:
                self.move_repeats += 1
            else:
                self.move_transitions += 1
        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        # Learn
        reward = self.get_score_delta(my_move, opponent_move)
        new_state = self.get_profiled_state()
        future = max(self.q_table[new_state].values())
        old_value = self.q_table[self.last_state][self.last_used]
        self.q_table[self.last_state][self.last_used] += self.learning_rate * (
            reward + self.discount_factor * future - old_value
        )

        # Infer opponent type
        self.predict_opponent_type()

    def predict_opponent_type(self):
        total = sum(self.move_counts.values())
        if total < 10:
            return  # not enough data

        bias = max(self.move_counts.values()) / total
        repeat_rate = self.move_repeats / total if total > 1 else 0.0
        switch_rate = self.move_transitions / total if total > 1 else 0.0

        if switch_rate > 0.6:
            self.predicted_type = 2  # Pattern-seeker
        elif repeat_rate > 0.5:
            self.predicted_type = 1  # Reactive or mimic
        elif bias < 0.4 and repeat_rate < 0.3:
            self.predicted_type = 3  # Stochastic or noise
        else:
            self.predicted_type = 0  # Unknown/ambiguous

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
        if len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2]:
            repeat = 1
        else:
            repeat = 0

        total = sum(self.move_counts.values())
        bias = max(self.move_counts.values()) / total if total else 0

        bias_level = 2 if bias > 0.6 else 1 if bias > 0.4 else 0
        switch = self.move_transitions / total if total > 1 else 0

        switch_bucket = 2 if switch > 0.6 else 1 if switch > 0.3 else 0
        return (repeat, bias_level, switch_bucket)
