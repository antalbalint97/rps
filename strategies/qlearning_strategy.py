import random

class QLearningStrategy:
    name = "QLearningStrategy"

    def __init__(self):
        self.q_table = {
            "rock": {"rock": 0.0, "paper": 0.0, "scissors": 0.0},
            "paper": {"rock": 0.0, "paper": 0.0, "scissors": 0.0},
            "scissors": {"rock": 0.0, "paper": 0.0, "scissors": 0.0},
            None: {"rock": 0.0, "paper": 0.0, "scissors": 0.0},  # Initial state
        }
        self.last_opponent_move = None  # state
        self.last_action = None         # action taken in that state

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def play(self) -> str:
        state = self.last_opponent_move

        if random.random() < self.epsilon:
            action = random.choice(["rock", "paper", "scissors"])
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_action = action
        return action

    def handle_moves(self, my_move: str, opponent_move: str):
        reward = self.get_reward(my_move, opponent_move)

        if self.last_opponent_move is not None and self.last_action is not None:
            prev_state = self.last_opponent_move
            action = self.last_action

            next_best = max(self.q_table[opponent_move].values())
            old_value = self.q_table[prev_state][action]

            self.q_table[prev_state][action] += self.learning_rate * (
                reward + self.discount_factor * next_best - old_value
            )

        self.last_opponent_move = opponent_move

    def get_reward(self, move1: str, move2: str) -> int:
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
