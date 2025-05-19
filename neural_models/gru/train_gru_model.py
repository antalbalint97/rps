# neural_models/gru/train_gru_model.py

import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "training_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "gru.pt")
WINDOW_SIZE = 4
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---
def one_hot_encode(index, num_classes=3):
    vec = np.zeros(num_classes)
    vec[index] = 1.0
    return vec

# --- MODEL ---
class GRURPSNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, output_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)

# --- DATASET ---
class RPSDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                flat_input = list(map(int, row[:-1]))
                target = int(row[-1])
                sequence = []
                for i in range(0, len(flat_input), 2):
                    my = one_hot_encode(flat_input[i])
                    opp = one_hot_encode(flat_input[i + 1])
                    sequence.append(np.concatenate([my, opp]))  # [6]
                self.samples.append((np.array(sequence), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- TRAIN ---
def train_model():
    dataset = RPSDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GRURPSNet(input_dim=6, hidden_dim=64, output_dim=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"GRU model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
