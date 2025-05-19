import random
from collections import defaultdict

class BayesianUpdateStrategy:
    name = "BayesianUpdateStrategy"

    def __init__(self):
        self.move_counts = defaultdict(int)
        self.total_moves = 0
        self.moves = ["rock", "paper", "scissors"]

    def play(self):
        if self.total_moves == 0:
            return random.choice(self.moves)

        # Estimate probabilities of opponent's next move
        probs = {
            move: self.move_counts[move] / self.total_moves
            for move in self.moves
        }

        # Calculate expected value of countering each move
        expected_rewards = {
            "rock": probs["scissors"],     # beats scissors
            "paper": probs["rock"],        # beats rock
            "scissors": probs["paper"]     # beats paper
        }

        # Choose the move with highest expected value
        best_move = max(expected_rewards, key=expected_rewards.get)
        return best_move

    def handle_moves(self, own_move, opponent_move):
        self.move_counts[opponent_move] += 1
        self.total_moves += 1
