import random
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class AntiMetaV5_OverfitPunisher:
    name = "AntiMetaV5_OverfitPunisher"

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
        self.my_history = []
        self.move_counts = Counter()
        self.repeat_windows = []

        self.strategy_usage = Counter()

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def play(self):
        state = self.get_profiled_state()

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_used = action
        self.last_state = state

        self.strategy_usage[action] += 1

        return self.strategies[action].play()

    def handle_moves(self, my_move, opponent_move):
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        self.opp_history.append(opponent_move)
        self.my_history.append(my_move)
        self.move_counts[opponent_move] += 1

        # Overfitting detection: are we being consistently countered?
        if len(self.my_history) >= 3:
            recent = self.my_history[-3:]
            counters = [self.counter_move(m) for m in recent]
            if opponent_move in counters:
                self.repeat_windows.append(1)
            else:
                self.repeat_windows.append(0)
            if len(self.repeat_windows) > 10:
                self.repeat_windows.pop(0)

        punished = sum(self.repeat_windows) > 6

        reward = self.get_score_delta(my_move, opponent_move)
        if punished:
            reward -= 1  # discourage repetition if being overfitted

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
        total = sum(self.move_counts.values())
        bias = max(self.move_counts.values()) / total if total else 0
        overfitted = int(sum(self.repeat_windows) > 6)
        bias_level = 2 if bias > 0.6 else 1 if bias > 0.4 else 0
        return (bias_level, overfitted)

    def counter_move(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
