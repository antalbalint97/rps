import random
from strategies.cycle_strategy import CycleStrategy
from strategies.metalearnerv6_strategy import MetaLearnerV6_AdaptiveQ

class AntiMetaV1_FalsePattern:
    name = "AntiMetaV1_FalsePattern"

    def __init__(self, inject_every=100, inject_duration=10):
        self.meta_strategy = MetaLearnerV6_AdaptiveQ()
        self.decoy_strategy = CycleStrategy()
        self.inject_every = inject_every
        self.inject_duration = inject_duration
        self.round_counter = 0
        self.in_decoy_mode = False
        self.decoy_rounds_left = 0

    def play(self):
        self.round_counter += 1

        # Start false pattern injection
        if not self.in_decoy_mode and self.round_counter % self.inject_every == 0:
            self.in_decoy_mode = True
            self.decoy_rounds_left = self.inject_duration

        if self.in_decoy_mode:
            self.decoy_rounds_left -= 1
            if self.decoy_rounds_left <= 0:
                self.in_decoy_mode = False
            return self.decoy_strategy.play()
        else:
            return self.meta_strategy.play()

    def handle_moves(self, my_move, opponent_move):
        self.meta_strategy.handle_moves(my_move, opponent_move)
        self.decoy_strategy.handle_moves(my_move, opponent_move)

    @property
    def strategy_usage(self):
        return self.meta_strategy.strategy_usage
