from strategies.cycle_strategy import CycleStrategy
from strategies.lastn_strategy import LastNStrategy
from strategies.frequency_strategy import FrequencyStrategy

class AdaptiveSwitcherStrategy:
    name = "AdaptiveSwitcherStrategy"

    def __init__(self):
        self.strategies = [CycleStrategy(), FrequencyStrategy(), LastNStrategy()]
        self.performance = [0] * len(self.strategies)
        self.current_index = 0
        self.history = []

    def play(self):
        return self.strategies[self.current_index].play()

    def handle_moves(self, my_move, opponent_move):
        for strat in self.strategies:
            strat.handle_moves(my_move, opponent_move)
        self.history.append((my_move, opponent_move))
        if len(self.history) > 10:
            my, opp = self.history[-1]
            if self._win(my, opp):
                self.performance[self.current_index] += 1
            else:
                self.performance[self.current_index] -= 1
            if sum(self.performance) < -3:
                self.current_index = (self.current_index + 1) % len(self.strategies)
                self.performance[self.current_index] = 0

    def _win(self, a, b):
        return (a == "rock" and b == "scissors") or \
               (a == "paper" and b == "rock") or \
               (a == "scissors" and b == "paper")
