from arena.game_engine import GameEngine
from strategies.random_strategy import RandomStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from strategies.cycle_strategy import CycleStrategy
from strategies.copycat_strategy import CopycatStrategy
from strategies.alwaysrock_strategy import AlwaysRockStrategy
from strategies.frequency_strategy import FrequencyStrategy
from strategies.lastn_strategy import LastNStrategy
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.mirrorbaiter_strategy import MirrorBaiterStrategy
from strategies.fmpnet_strategy import FMPNetStrategy
from strategies.delayedmirror_strategy import DelayedMirrorStrategy
from strategies.noisecounter_strategy import NoiseCounterStrategy
from strategies.entropymax_strategy import EntropyMaximizerStrategy

def main():
    engine = GameEngine()
    strategies = [
        RandomStrategy,
        EnhancedStrategy,
        CycleStrategy,
        CopycatStrategy,
        AlwaysRockStrategy,
        FrequencyStrategy,
        LastNStrategy,
        MarkovStrategy,
        QLearningStrategy,
        MirrorBaiterStrategy,
        FMPNetStrategy,
        DelayedMirrorStrategy,
        NoiseCounterStrategy,
        EntropyMaximizerStrategy
    ]
    
    tournament_data = engine.round_robin_tournament(strategies)
    results = tournament_data["final_scores"]

    print("\n--- Final Scores ---")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {score:.2f} average points")

    return tournament_data

if __name__ == "__main__":
    main()