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
from strategies.delayedmirror_strategy import DelayedMirrorStrategy
from strategies.noisecounter_strategy import NoiseCounterStrategy
from strategies.entropymax_strategy import EntropyMaximizerStrategy
from strategies.adaptiveswitcher_strategy import AdaptiveSwitcherStrategy
from strategies.patternhunter_strategy import PatternHunterStrategy
from strategies.bayesian_strategy import BayesianNGramStrategy
# from strategies.fmpnet_strategy import FMPNetStrategy
# from strategies.lstm_strategy import LSTMStrategy
# from strategies.gru_strategy import GRUStrategy
from strategies.thompsonmeta_strategy import ThompsonMetaStrategy
from strategies.thompsonmetav2_strategy import ThompsonMetaV2
from strategies.thompsonmetav3_strategy import ThompsonMetaV3_Contextual
from strategies.boltzmann_strategy import BoltzmannMetaStrategy
from strategies.thompsonmetav4_strategy import ThompsonMetaV4_Profiled
from strategies.metalearnerv5_strategy import MetaLearnerV5_QController
from strategies.metalearnerv6_strategy import MetaLearnerV6_AdaptiveQ
from strategies.antimetav1_strategy import AntiMetaV1_FalsePattern
from strategies.antimetav2_strategy import AntiMetaV2_MetaPredictor
from strategies.antimetav3_strategy import AntiMetaV3_Deprivation
from strategies.antimetav4_strategy import AntiMetaV4_MirrorDiverge
from strategies.antimetav5_strategy import AntiMetaV5_OverfitPunisher
from strategies.metalearnerv7_strategy import MetaLearnerV7_ShadowQ
from strategies.metalearnerv8_strategy import MetaLearnerV8_HybridDeceptiveQ
from strategies.dreamweaverv7_strategy import DreamWeaverV7
from strategies.shinydiamond_strategy import ShinyDiamond



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
        DelayedMirrorStrategy,
        NoiseCounterStrategy,
        EntropyMaximizerStrategy,
        AdaptiveSwitcherStrategy,
        PatternHunterStrategy,
        BayesianNGramStrategy,
        ThompsonMetaStrategy,
        ThompsonMetaV2,
        ThompsonMetaV3_Contextual,
        BoltzmannMetaStrategy,
        ThompsonMetaV4_Profiled,
        MetaLearnerV5_QController,
        MetaLearnerV6_AdaptiveQ,
        AntiMetaV1_FalsePattern,
        AntiMetaV2_MetaPredictor,
        AntiMetaV3_Deprivation,
        AntiMetaV4_MirrorDiverge,
        AntiMetaV5_OverfitPunisher,
        MetaLearnerV7_ShadowQ,
        MetaLearnerV8_HybridDeceptiveQ,
        DreamWeaverV7,
        ShinyDiamond
    ]
    
    tournament_data = engine.round_robin_tournament(strategies)
    results = tournament_data["final_scores"]

    print("\n--- Final Scores ---")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {score:.2f} average points")

    return tournament_data

if __name__ == "__main__":
    main()