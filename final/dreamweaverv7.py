from rpsa_sdk.strategy import Strategy
from rpsa_sdk.helpers import counter_move
import random
import math
from collections import defaultdict, Counter


class DreamWeaverV7(Strategy):
    name = "DreamWeaverV7"

    def __init__(self, model=None):
        super().__init__(model)

        self.strategies = {
            "markov": SecondOrderMarkov(),
            "enhanced": EnhancedStrategyDelay(),
            "qlearn": QLearningStrategyV2()
        }
        self.actions = list(self.strategies.keys())

        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.last_sa = None

        self.opp_history = []
        self.move_counts = Counter()
        self.strategy_scores = defaultdict(int)
        self.strategy_use = defaultdict(int)
        self.meta_state = None

        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.epsilon = 0.1

        self.false_pattern_active = False
        self.false_pattern_sequence = ["rock", "rock", "rock"]
        self.false_pattern_index = 0
        self.false_pattern_triggered = False
        self.loss_streak = 0

    def play(self):
        state = self.get_profiled_state()

        # False pattern injection logic
        if self.false_pattern_triggered:
            move = self.false_pattern_sequence[self.false_pattern_index % len(self.false_pattern_sequence)]
            self.false_pattern_index += 1
            if self.false_pattern_index >= len(self.false_pattern_sequence):
                self.false_pattern_triggered = False  # end baiting
            return move

        action = self.select_action(state)
        self.last_sa = (state, action)
        self.meta_state = action
        return self.strategies[action].play()

    def select_action(self, state):
        if self.loss_streak >= 3 and not self.false_pattern_triggered:
            self.false_pattern_triggered = True
            self.false_pattern_index = 0
            return "rock"

        if self.opponent_entropy() < 1.2 and random.random() < 0.5:
            return "enhanced"

        if self.opponent_entropy() > 1.6 and random.random() < 0.3:
            return random.choice(["markov", "qlearn"])

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return max(self.q_table[state], key=self.q_table[state].get)

    def opponent_entropy(self):
        count = Counter(self.opp_history[-20:])
        total = sum(count.values())
        if total == 0:
            return 2.0
        return -sum((freq / total) * math.log2(freq / total) for freq in count.values())

    def handle_moves(self, my_move, opponent_move):
        for strat in self.strategies.values():
            strat.handle_moves(my_move, opponent_move)

        self.opp_history.append(opponent_move)
        self.move_counts[opponent_move] += 1

        score = self.get_score_delta(my_move, opponent_move)
        if score == -1:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        if self.last_sa:
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


# --- Sub-strategies ---

class SecondOrderMarkov:
    name = "SecondOrderMarkov"

    def __init__(self, order=2, alpha=0.2):
        self.order = order
        self.alpha = alpha
        self.history = []
        self.transition_probs = defaultdict(lambda: defaultdict(float))

    def play(self):
        if len(self.history) < self.order:
            return random.choice(["rock", "paper", "scissors"])
        key = tuple(self.history[-self.order:])
        next_probs = self.transition_probs.get(key, {})
        if not next_probs:
            return random.choice(["rock", "paper", "scissors"])
        predicted = max(next_probs, key=next_probs.get)
        return counter_move(predicted)

    def handle_moves(self, own_move, opponent_move):
        if len(self.history) >= self.order:
            key = tuple(self.history[-self.order:])
            for move in ["rock", "paper", "scissors"]:
                observed = 1.0 if move == opponent_move else 0.0
                old = self.transition_probs[key][move]
                self.transition_probs[key][move] = (1 - self.alpha) * old + self.alpha * observed
        self.history.append(opponent_move)


class EnhancedStrategyDelay:
    name = "EnhancedStrategy"

    def __init__(self, delay=0, response_chance=0.8):
        self.history = []
        self.delay = delay
        self.response_chance = response_chance

    def play(self):
        if len(self.history) <= self.delay or random.random() > self.response_chance:
            return random.choice(["rock", "paper", "scissors"])
        delayed_index = -1 - self.delay
        move_to_counter = self.history[delayed_index][1]
        return counter_move(move_to_counter)

    def handle_moves(self, own_move, opponent_move):
        self.history.append((own_move, opponent_move))


class QLearningStrategyV2:
    name = "QLearningStrategy"

    def __init__(self):
        self.q_table = defaultdict(lambda: {"rock": 0.0, "paper": 0.0, "scissors": 0.0})
        self.last_state = None
        self.last_action = None

        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.epsilon = 0.2
        self.epsilon_decay = 0.99
        self.temperature = 0.5
        self.decay_rate = 0.01

        self.last_opponent_move = None
        self.last_my_move = None

    def play(self):
        state = (self.last_opponent_move, self.last_my_move)
        self._ensure_state(state)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def select_action(self, state):
        q_values = self.q_table[state]
        exp_vals = {a: math.exp(q / self.temperature) for a, q in q_values.items()}
        total = sum(exp_vals.values())
        probs = {a: v / total for a, v in exp_vals.items()}
        return random.choices(list(probs.keys()), weights=probs.values())[0]

    def handle_moves(self, my_move, opponent_move):
        reward = self.get_result(my_move, opponent_move)
        new_state = (opponent_move, my_move)
        self._ensure_state(new_state)

        for action in self.q_table[self.last_state]:
            self.q_table[self.last_state][action] *= (1 - self.decay_rate)

        max_future_q = max(self.q_table[new_state].values())
        old_q = self.q_table[self.last_state][self.last_action]

        self.q_table[self.last_state][self.last_action] = (
            (1 - self.learning_rate) * old_q +
            self.learning_rate * (reward + self.discount_factor * max_future_q)
        )

        self.last_opponent_move = opponent_move
        self.last_my_move = my_move
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def get_result(self, move1, move2):
        if move1 == move2:
            return 0
        elif (move1 == "rock" and move2 == "scissors") or \
             (move1 == "scissors" and move2 == "paper") or \
             (move1 == "paper" and move2 == "rock"):
            return 1
        else:
            return -1

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {"rock": 0.0, "paper": 0.0, "scissors": 0.0}


strategy = DreamWeaverV7
