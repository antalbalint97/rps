import random
from collections import defaultdict, Counter
from strategies.markov_strategy import MarkovStrategy
from strategies.qlearning_strategy import QLearningStrategy
from strategies.enhanced_strategy import EnhancedStrategy

class AntiMetaV3_Deprivation:
    name = "AntiMetaV3_Deprivation"

    def __init__(self):
        self.strategies = {
            "markov": MarkovStrategy(),
            "enhanced": EnhancedStrategy(),
            "qlearn": QLearningStrategy()
        }
        self.actions = list(self.strategies.keys())
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})

        self.last_used = None
        self.last_state = None

        self.opp_history = []
        self.move_counts = Counter()
        self.move_repeats = 0
        self.move_transitions = 0

        self.strategy_usage = Counter()

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

        # Noise config
        self.deprivation_threshold = 50  # Start injecting noise after N rounds
        self.noise_frequency = 5  # Every Nth round add noise
        self.round_counter = 0

    def play(self):
        self.round_counter += 1
        state = self.get_profiled_state()

        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        self.last_used = action
        self.last_state = state
        self.strategy_usage[action] += 1

        move = self.strategies[action].play()

        # Information deprivation: randomize every nth move
        if self.round_counter > self.deprivation_threshold and self.round_counter % self.noise_frequency == 0:
            return random.choice(["rock", "paper", "scissors"])

        return move

    def handle_moves(self, my_move, opponent_move):
        for strategy in self.strategies.values():
            strategy.handle_moves(my_move, opponent_move)

        if self.opp_history:
            if opponent_move == self.opp_history[-1]:
                self.move_repeats += 1
            else:
                self.move_transitions += 1

        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        reward = self.get_score_delta(my_move, opponent_move)
        new_state = self.get_profiled_state()
        future = max(self.q_table[new_state].values())
        old_value = self.q_table[self.last_state][self.last_used]
        self.q_table[self.last_state][self.last_used] += self.learning_rate * (
            reward + self.discount_factor * future - old_value
        )

    def get_score_delta(self, move1, move2):
        if move1 == move2:
            return 0
        elif (
            (move1 == "rock" and move2 == "scissors") or
            (move1 == "scissors" and move2 == "paper") or
            (move1 == "paper" and move2 == "rock")
        ):
            return 1
        else:
            return -1

    def get_profiled_state(self):
        if len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2]:
            repeat = 1
        else:
            repeat = 0

        total = sum(self.move_counts.values())
        bias = max(self.move_counts.values()) / total if total else 0

        bias_level = 2 if bias > 0.6 else 1 if bias > 0.4 else 0
        switch = self.move_transitions / total if total > 1 else 0

        switch_bucket = 2 if switch > 0.6 else 1 if switch > 0.3 else 0
        return (repeat, bias_level, switch_bucket)
