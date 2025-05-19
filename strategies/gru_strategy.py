# strategies/gru_strategy.py

import os
import torch
import torch.nn as nn
import random
from collections import deque

# Define the GRU model (must match training architecture)
class GRURPSNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, output_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


# Strategy wrapper
class GRUStrategy:
    name = "GRUStrategy"

    def __init__(self):
        model_path = os.environ.get("MODEL_PATH", "neural_models/gru/gru.pt")

        self.model = GRURPSNet()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

        self.history = deque(maxlen=4)
        self.move_to_idx = {"rock": 0, "paper": 1, "scissors": 2}
        self.idx_to_move = ["rock", "paper", "scissors"]

    def play(self):
        if len(self.history) < 4:
            return random.choice(["rock", "paper", "scissors"])

        input_vec = []
        for my, opp in self.history:
            input_vec.extend([
                float(my == "rock"), float(my == "paper"), float(my == "scissors"),
                float(opp == "rock"), float(opp == "paper"), float(opp == "scissors")
            ])

        input_tensor = torch.tensor(input_vec, dtype=torch.float32).reshape(1, 4, 6)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            return self.counter(self.idx_to_move[predicted_idx])

    def handle_moves(self, my_move, opponent_move):
        self.history.append((my_move, opponent_move))

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
