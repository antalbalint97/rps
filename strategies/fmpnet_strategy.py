import os
import torch
import random
from collections import deque
import torch.nn as nn

# Define model architecture
class RPSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)

# Strategy wrapper
class FMPNetStrategy:
    name = "FMPNetStrategy"

    def __init__(self):
        model_path = os.environ.get("MODEL_PATH", "neural_models/fmpnet/fmpnet.pt")

        self.model = self.load_model(model_path)
        self.model.eval()

        self.history = deque(maxlen=4)
        self.move_to_idx = {"rock": 0, "paper": 1, "scissors": 2}
        self.idx_to_move = ["rock", "paper", "scissors"]

    def load_model(self, path):
        model = RPSNet()
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        return model

    def play(self):
        if len(self.history) < 4:
            return random.choice(["rock", "paper", "scissors"])

        input_vec = []
        for my, opp in self.history:
            input_vec.append(self.move_to_idx[my])
            input_vec.append(self.move_to_idx[opp])

        input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
            return self.counter(self.idx_to_move[predicted])

    def handle_moves(self, my_move, opponent_move):
        self.history.append((my_move, opponent_move))

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
