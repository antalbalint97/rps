import random
import math
from collections import deque

class MCTSStrategyUCB:
    name = "MCTSStrategyUCB"

    def __init__(self, total_playouts=5, rollout_depth=2, exploration_factor=1.41):
        self.total_playouts = total_playouts
        self.rollout_depth = rollout_depth
        self.exploration_factor = exploration_factor
        self.moves = ["rock", "paper", "scissors"]
        self.history = deque(maxlen=10)

        # Tracking stats
        self.move_counts = {move: 0 for move in self.moves}
        self.move_rewards = {move: 0 for move in self.moves}

    def play(self):
        # Reset counts for this decision
        self.move_counts = {move: 0 for move in self.moves}
        self.move_rewards = {move: 0 for move in self.moves}

        for _ in range(self.total_playouts):
            move = self.select_ucb_move()
            reward = self.simulate_playout(move)
            self.move_counts[move] += 1
            self.move_rewards[move] += reward

        avg_scores = {
            move: self.move_rewards[move] / self.move_counts[move]
            if self.move_counts[move] > 0 else float("-inf")
            for move in self.moves
        }

        return max(avg_scores, key=avg_scores.get)

    def select_ucb_move(self):
        total_simulations = sum(self.move_counts.values()) + 1  # avoid log(0)
        best_ucb = float("-inf")
        best_move = None

        for move in self.moves:
            n_i = self.move_counts[move]
            w_i = self.move_rewards[move]

            if n_i == 0:
                # Try every move at least once
                return move

            avg_reward = w_i / n_i
            ucb = avg_reward + self.exploration_factor * math.sqrt(
                math.log(total_simulations) / n_i
            )

            if ucb > best_ucb:
                best_ucb = ucb
                best_move = move

        return best_move

    def simulate_playout(self, initial_move):
        my_score = 0
        my_move = initial_move
        for _ in range(self.rollout_depth):
            opp_move = random.choice(self.moves)
            my_score += self.score_round(my_move, opp_move)
            my_move = random.choice(self.moves)  # can enhance with learned behavior
        return my_score

    def score_round(self, my_move, opp_move):
        if my_move == opp_move:
            return 0
        elif (my_move == "rock" and opp_move == "scissors") or \
             (my_move == "scissors" and opp_move == "paper") or \
             (my_move == "paper" and opp_move == "rock"):
            return 1
        else:
            return -1

    def handle_moves(self, own_move, opponent_move):
        self.history.append((own_move, opponent_move))
