import random

class MirrorBaiterStrategy:
    name = "MirrorBaiterStrategy"

    def __init__(self):
        self.my_last_move = None
        self.opponent_last_move = None
        self.detected_mirroring = False
        self.bait_phase = False

    def play(self):
        if self.detected_mirroring:
            if self.bait_phase:
                # Deliver the counter to the bait they will now mirror
                return self.counter(self.my_last_move)
            else:
                # Play bait move — which they are likely to mirror next
                bait = self.bait_move()
                self.bait_phase = True
                return bait
        else:
            # Before detection, play Enhanced-style (counter opponent's last move)
            if self.opponent_last_move:
                return self.counter(self.opponent_last_move)
            else:
                return random.choice(["rock", "paper", "scissors"])

    def handle_moves(self, my_move, opponent_move):
        if self.my_last_move and opponent_move == self.my_last_move:
            self.detected_mirroring = True
        else:
            self.detected_mirroring = False
            self.bait_phase = False  # reset if no mirror behavior

        self.my_last_move = my_move
        self.opponent_last_move = opponent_move

    def counter(self, move):
        return {
            "rock": "paper",
            "paper": "scissors",
            "scissors": "rock"
        }[move]

    def bait_move(self):
        # Choose a move that we want them to copy so we can trap it
        # e.g., bait "rock" so we can play "paper" next
        return random.choice(["rock", "paper", "scissors"])
