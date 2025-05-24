import time
import random
import sys
import os
# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from strategies.freebird_strategy import FreeBird
from strategies.shinydiamond_strategy import ShinyDiamond
from strategies.toxic_strategy import Toxic
# from strategies.qlearningv2_strategy import QLearningStrategyV2
# from strategies.enhanceddelay_strategy import EnhancedStrategyDelay
# from strategies.secondordermarkov_strategy import SecondOrderMarkov


def benchmark_strategy(strategy_class, rounds=2000):
    strategy = strategy_class()
    total_time = 0.0
    my_move = "rock"
    opponent_move = "rock"

    for _ in range(rounds):
        start = time.perf_counter()
        move = strategy.play()
        end = time.perf_counter()
        total_time += (end - start)
        strategy.handle_moves(my_move, opponent_move)
        my_move = move
        opponent_move = random.choice(["rock", "paper", "scissors"])

    print(f"{strategy_class.__name__:<25} Total Time: {total_time:.4f}s {'✅' if total_time < 0.1 else '❌'}")

if __name__ == "__main__":
    # benchmark_strategy(ShinyDiamond)
    # benchmark_strategy(QLearningStrategyV2)
    # benchmark_strategy(EnhancedStrategyDelay)
    # benchmark_strategy(SecondOrderMarkov)
    # benchmark_strategy(FreeBird)
    benchmark_strategy(Toxic)
