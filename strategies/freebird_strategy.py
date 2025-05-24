import random
import math
from collections import defaultdict, Counter, deque
from strategies.secondordermarkov_strategy import SecondOrderMarkov
from strategies.qlearningv2_strategy import QLearningStrategyV2
from strategies.enhanceddelay_strategy import EnhancedStrategyDelay
from strategies.bayesian_strategy import BayesianNGramStrategy
from strategies.mcts_strategy import MCTSStrategyUCB
from strategies.noiseinjection_strategy import NoiseInjectionStrategy


class FreeBird:
    name = "FreeBird"

    def __init__(self, model=None):
        self.strategies = {
            "markov": SecondOrderMarkov(),
            "enhanced": EnhancedStrategyDelay(),
            "qlearn": QLearningStrategyV2(),
            "bayesian": BayesianNGramStrategy(),
            "mcts": MCTSStrategyUCB(),
            "noise": NoiseInjectionStrategy()
        }
        self.actions = list(self.strategies.keys())

        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.last_sa = None

        self.opp_history = []
        self.move_counts = Counter()
        self.strategy_streaks = defaultdict(int)
        self.current_streak_strategy = None

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def play(self):
        state = self.get_profiled_state()
        action_votes = defaultdict(float)

        for name, strategy in self.strategies.items():
            move = strategy.play()
            weight = 1.0
            if name == "bayesian" and hasattr(strategy, "_get_confidence"):
                context = tuple(strategy.history)
                if context in strategy.context_counts:
                    probs = strategy._compute_smoothed_probs(context)
                    weight = strategy._get_confidence(probs)
            action_votes[move] += weight

        final_move = max(action_votes, key=action_votes.get)

        action = self.select_action(state)
        if action == self.current_streak_strategy:
            self.strategy_streaks[action] += 1
        else:
            if self.current_streak_strategy:
                self.strategy_streaks[self.current_streak_strategy] = 0
            self.current_streak_strategy = action
            self.strategy_streaks[action] = 1

        self.last_sa = (state, action)
        return final_move

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        scores = self.q_table[state].copy()
        for action in scores:
            scores[action] -= self.strategy_streaks[action] * 0.1

        return max(scores, key=scores.get)

    def handle_moves(self, my_move, opponent_move):
        for strat in self.strategies.values():
            strat.handle_moves(my_move, opponent_move)

        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        score = self.get_score_delta(my_move, opponent_move)
        if self.last_sa and self.last_sa[1] in self.actions:
            state, action = self.last_sa
            new_state = self.get_profiled_state()

            future = max(self.q_table[new_state].values())
            old_value = self.q_table[state][action]
            self.q_table[state][action] += self.learning_rate * (
                score + self.discount_factor * future - old_value
            )

        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_score_delta(self, move1, move2):
        if move1 == move2:
            return 0
        elif (move1 == "rock" and move2 == "scissors") or \
             (move1 == "scissors" and move2 == "paper") or \
             (move1 == "paper" and move2 == "rock"):
            return 1
        else:
            return -1

    def get_profiled_state(self):
        recent_repeat = int(len(self.opp_history) >= 2 and self.opp_history[-1] == self.opp_history[-2])
        total = sum(self.move_counts.values())
        bias_level = 0
        switch_rate = 0

        if total:
            bias_ratio = max(self.move_counts.values()) / total
            if bias_ratio > 0.6:
                bias_level = 2
            elif bias_ratio > 0.4:
                bias_level = 1
            switch_rate = sum(self.opp_history[i] != self.opp_history[i - 1] for i in range(1, len(self.opp_history))) / total

        switch_bucket = 2 if switch_rate > 0.6 else 1 if switch_rate > 0.3 else 0
        return (recent_repeat, bias_level, switch_bucket)