import random
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class MetaLearnerV5_QController:
    name = "MetaLearnerV5_QController"

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
        self.opp_history = []
        self.move_counts = Counter()

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1  # exploration rate

    def play(self):
        state = self.get_state()

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            action = max(q_values, key=q_values.get)

        self.last_used = action
        self.last_state = state
        return self.strategies[action].play()

    def handle_moves(self, my_move, opponent_move):
        # Update sub-strategies
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        # Update move history and counts
        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        # Get reward
        reward = self.get_score_delta(my_move, opponent_move)

        # Update Q-table
        new_state = self.get_state()
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

    def get_state(self):
        # Opponent repeated last move?
        recent_repeat = 0
        if len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2]:
            recent_repeat = 1

        # Move bias level
        total = sum(self.move_counts.values())
        if total == 0:
            bias_level = 0
        else:
            most_common_ratio = max(self.move_counts.values()) / total
            if most_common_ratio > 0.6:
                bias_level = 2
            elif most_common_ratio > 0.4:
                bias_level = 1
            else:
                bias_level = 0

        return (recent_repeat, bias_level)
