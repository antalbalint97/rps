import os
import torch
import random
import torch.nn as nn
from collections import deque

# --- Re-define the model architecture ---
class LSTMRPSNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- Strategy class ---
class LSTMStrategy:
    name = "LSTMStrategy"

    def __init__(self):
        model_path = os.environ.get("MODEL_PATH", "neural_models/lstm/lstm.pt")

        # Instantiate and load weights
        self.model = LSTMRPSNet()
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
            predicted = torch.argmax(output, dim=1).item()
            return self.counter(self.idx_to_move[predicted])

    def handle_moves(self, my_move, opponent_move):
        self.history.append((my_move, opponent_move))

    def counter(self, move):
        return {"rock": "paper", "paper": "scissors", "scissors": "rock"}[move]
