class EnhancedStrategy:
    name = "EnhancedStrategy"

    def __init__(self):
        self.history = []

    def play(self) -> str:
        if not self.history:
            return "rock"

        last_opponent_move = self.history[-1][1]
        return self.counter_move(last_opponent_move)

    def handle_moves(self, own_move: str, opponent_move: str):
        self.history.append((own_move, opponent_move))

    def counter_move(self, move: str) -> str:
        counter = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
        return counter.get(move, "rock")
