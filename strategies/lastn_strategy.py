class LastNStrategy:
    name = "LastNStrategy"

    def __init__(self, n=3):
        self.memory = []
        self.n = n

    def play(self):
        if not self.memory:
            return "rock"
        window = self.memory[-self.n:]
        most_common = max(set(window), key=window.count)
        return self.counter(most_common)

    def handle_moves(self, my_move, opponent_move):
        self.memory.append(opponent_move)

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
