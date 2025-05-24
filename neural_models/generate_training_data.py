import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import csv
from collections import deque
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


# Settings
NUM_PLAYS_PER_GAME = 2000
NUM_GAMES_PER_MATCH = 100
WINDOW_SIZE = 4

# Define relative output path inside neural_models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "training_data.csv")

move_to_idx = {"rock": 0, "paper": 1, "scissors": 2}

strategies = [
    RandomStrategy,                  # For baseline unpredictability
    CycleStrategy,                   # For loop pattern detection
    CopycatStrategy,                 # Mirrors your moves
    AlwaysRockStrategy,              # Exploitable fixed behavior
    FrequencyStrategy,               # Exploits your frequent plays
    LastNStrategy,                   # Recent memory based
    MarkovStrategy,                  # Conditional transition patterns
    QLearningStrategy,               # Mildly adaptive, value-based
    MirrorBaiterStrategy,            # Baits mirroring patterns
    DelayedMirrorStrategy,          # Reacts with a delay
    NoiseCounterStrategy,           # Noisy but patterned
    AdaptiveSwitcherStrategy,       # Switches between strategies
    PatternHunterStrategy,          # Detects and reacts to patterns
    BayesianNGramStrategy           # Learns conditional sequences
]

engine = GameEngine()
engine.NUM_PLAYS_PER_GAME = NUM_PLAYS_PER_GAME
engine.NUM_GAMES_PER_MATCH = NUM_GAMES_PER_MATCH

dataset = []

for i in range(len(strategies)):
    for j in range(i + 1, len(strategies)):
        s1_cls = strategies[i]
        s2_cls = strategies[j]

        for _ in range(NUM_GAMES_PER_MATCH):
            s1 = s1_cls()
            s2 = s2_cls()
            history = deque(maxlen=WINDOW_SIZE)

            for _ in range(NUM_PLAYS_PER_GAME):
                move1 = s1.play()
                move2 = s2.play()

                if len(history) == WINDOW_SIZE:
                    row = []
                    for prev_move1, prev_move2 in history:
                        row.extend([move_to_idx[prev_move1], move_to_idx[prev_move2]])
                    row.append(move_to_idx[move2])  # Target: opponent's move
                    dataset.append(row)

                history.append((move1, move2))
                s1.handle_moves(move1, move2)
                s2.handle_moves(move2, move1)

# Save dataset
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"m{i}" for i in range(WINDOW_SIZE * 2)] + ["target"]
    writer.writerow(header)
    writer.writerows(dataset)

print(f"Dataset saved to {OUTPUT_CSV} with {len(dataset):,} samples.")
