import random

class RandomStrategy:
    name = "RandomStrategy"

    def play(self) -> str:
        return random.choice(["rock", "paper", "scissors"])

    def handle_moves(self, own_move: str, opponent_move: str):
        pass  # Doesn't adapt or learn
