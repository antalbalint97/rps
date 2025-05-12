import random

class QLearningStrategy:
    name = "QLearningStrategy"

    def __init__(self):
        self.q_table = {
            "rock": {"rock": 0, "paper": 0, "scissors": 0},
            "paper": {"rock": 0, "paper": 0, "scissors": 0},
            "scissors": {"rock": 0, "paper": 0, "scissors": 0},
            None: {"rock": 0, "paper": 0, "scissors": 0},  # For first move
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
        elif (
            (move1 == "rock" and move2 == "scissors") or
            (move1 == "scissors" and move2 == "paper") or
            (move1 == "paper" and move2 == "rock")
        ):
            return 1
        else:
            return -1
