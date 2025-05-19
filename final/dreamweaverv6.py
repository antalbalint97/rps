from rpsa_sdk.strategy import Strategy
from rpsa_sdk.helpers import counter_move
import random
from collections import defaultdict, Counter


class DreamWeaverV6(Strategy):
    name = "DreamWeaverV6"

    def __init__(self, model=None):
        super().__init__(model)

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
        self.move_repeats = 0
        self.move_transitions = 0

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

        return self.strategies[action].play()

    def handle_moves(self, my_move, opponent_move):
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        if self.opp_history and opponent_move == self.opp_history[-1]:
            self.move_repeats += 1
        elif self.opp_history:
            self.move_transitions += 1

        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        reward = self.get_score_delta(my_move, opponent_move)

        new_state = self.get_profiled_state()
        future = max(self.q_table[new_state].values())
        old_value = self.q_table[self.last_state][self.last_used]

        self.q_table[self.last_state][self.last_used] += self.learning_rate * (
            reward + self.discount_factor * future - old_value
        )

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


# --- Sub-strategies used by DreamWeaverV6 ---

class MarkovStrategy:
    name = "MarkovStrategy"

    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_move = None

    def play(self):
        if self.last_move is None:
            return random.choice(["rock", "paper", "scissors"])
        next_probs = self.transition_counts[self.last_move]
        if not next_probs:
            return random.choice(["rock", "paper", "scissors"])
        predicted = max(next_probs, key=next_probs.get)
        return self.counter(predicted)

    def handle_moves(self, my_move, opponent_move):
        if self.last_move is not None:
            self.transition_counts[self.last_move][opponent_move] += 1
        self.last_move = opponent_move

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]


class EnhancedStrategy:
    name = "EnhancedStrategy"

    def __init__(self):
        self.history = []

    def play(self):
        if not self.history:
            return "rock"
        last_opponent_move = self.history[-1][1]
        return self.counter_move(last_opponent_move)

    def handle_moves(self, own_move, opponent_move):
        self.history.append((own_move, opponent_move))

    def counter_move(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}.get(move, "rock")


class QLearningStrategy:
    name = "QLearningStrategy"

    def __init__(self):
        self.q_table = {
            "rock": {"rock": 0, "paper": 0, "scissors": 0},
            "paper": {"rock": 0, "paper": 0, "scissors": 0},
            "scissors": {"rock": 0, "paper": 0, "scissors": 0},
            None: {"rock": 0, "paper": 0, "scissors": 0},
        }
        self.last_opponent_move = None
        self.last_action = None
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def play(self):
        state = self.last_opponent_move
        if random.random() < self.epsilon:
            action = random.choice(["rock", "paper", "scissors"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        self.last_action = action
        return action

    def handle_moves(self, my_move, opponent_move):
        result = self.get_result(my_move, opponent_move)
        reward = {1: 1, 0: 0, -1: -1}[result]

        if self.last_opponent_move is not None:
            prev_state = self.last_opponent_move
            next_best = max(self.q_table[opponent_move].values())
            self.q_table[prev_state][my_move] += self.learning_rate * (
                reward + self.discount_factor * next_best - self.q_table[prev_state][my_move]
            )

        self.last_opponent_move = opponent_move

    def get_result(self, move1, move2):
        if move1 == move2:
            return 0
        elif (move1 == "rock" and move2 == "scissors") or \
             (move1 == "scissors" and move2 == "paper") or \
             (move1 == "paper" and move2 == "rock"):
            return 1
        else:
            return -1

strategy = DreamWeaverV6
