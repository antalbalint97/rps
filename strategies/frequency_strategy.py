class FrequencyStrategy:
    name = "FrequencyStrategy"

    def __init__(self):
        self.opponent_moves = []

    def play(self):
        if not self.opponent_moves:
            return "rock"
        most_common = max(set(self.opponent_moves), key=self.opponent_moves.count)
        return self.counter(most_common)

    def handle_moves(self, my_move, opponent_move):
        self.opponent_moves.append(opponent_move)

    def counter(self, move):
        return {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }[move]
