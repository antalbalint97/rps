
import random
import math
from collections import defaultdict, Counter

class QLearningStrategyV2:
    name = "QLearningStrategy"

    def __init__(self):
        self.q_table = defaultdict(lambda: {"rock": 0.0, "paper": 0.0, "scissors": 0.0})
        self.last_state = None
        self.last_action = None

        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.epsilon = 0.2
        self.epsilon_decay = 0.99
        self.temperature = 0.5
        self.decay_rate = 0.01

        self.last_opponent_move = None
        self.last_my_move = None

    def play(self):
        state = (self.last_opponent_move, self.last_my_move)
        self._ensure_state(state)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def select_action(self, state):
        q_values = self.q_table[state]
        exp_vals = {a: math.exp(q / self.temperature) for a, q in q_values.items()}
        total = sum(exp_vals.values())
        probs = {a: v / total for a, v in exp_vals.items()}
        return random.choices(list(probs.keys()), weights=probs.values())[0]

    def handle_moves(self, my_move, opponent_move):
        reward = self.get_result(my_move, opponent_move)
        new_state = (opponent_move, my_move)
        self._ensure_state(new_state)

        if self.last_state is not None and self.last_action is not None:
            self._ensure_state(self.last_state)

            for action in self.q_table[self.last_state]:
                self.q_table[self.last_state][action] *= (1 - self.decay_rate)

            max_future_q = max(self.q_table[new_state].values())
            old_q = self.q_table[self.last_state][self.last_action]

            self.q_table[self.last_state][self.last_action] = (
                (1 - self.learning_rate) * old_q +
                self.learning_rate * (reward + self.discount_factor * max_future_q)
            )

        self.last_opponent_move = opponent_move
        self.last_my_move = my_move
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def get_result(self, move1, move2):
        if move1 == move2:
            return 0
        elif (move1 == "rock" and move2 == "scissors") or \
             (move1 == "scissors" and move2 == "paper") or \
             (move1 == "paper" and move2 == "rock"):
            return 1
        else:
            return -1

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {"rock": 0.0, "paper": 0.0, "scissors": 0.0}