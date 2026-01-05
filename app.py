import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load and Encode Data
# Neural networks require numerical indices, not raw IDs.
df = pd.read_csv("listening_history.csv")

user2idx = {u: i for i, u in enumerate(df['user_id'].unique())}
song2idx = {s: i for i, s in enumerate(df['song_id'].unique())}

df['user_idx'] = df['user_id'].map(user2idx)
df['song_idx'] = df['song_id'].map(song2idx)

num_users = len(user2idx)
num_songs = len(song2idx)

# Train / Test Split
train_df, test_df = train_test_split(
    df[['user_idx', 'song_idx', 'listen']],
    test_size=0.2,
    random_state=42
)

# PyTorch Dataset Class
class NCFDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_idx'].values, dtype=torch.long)
        self.songs = torch.tensor(data['song_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(data['listen'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.labels[idx]
    
train_loader = DataLoader(NCFDataset(train_df), batch_size=256, shuffle=True)
test_loader = DataLoader(NCFDataset(test_df), batch_size=256)

# Neural Collaborative Filtering Model
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_songs, emb_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.song_emb = nn.Embedding(num_songs, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user, song):
        u = self.user_emb(user)
        s = self.song_emb(song)
        x = torch.cat([u, s], dim=1)
        return self.mlp(x).squeeze()

# Training Setup    
device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralCF(num_users, num_songs).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for users, songs, labels in train_loader:
        users, songs, labels = users.to(device), songs.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(users, songs)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation: Precision@K
def precision_at_k(model, data_loader, k=10):
    model.eval()
    user_scores = {}

    with torch.no_grad():
        for users, songs, labels in data_loader:
            users, songs = users.to(device), songs.to(device)
            scores = model(users, songs)

            for u, s, score, label in zip(users, songs, scores, labels):
                u = u.item()
                if u not in user_scores:
                    user_scores[u] = []
                user_scores[u].append((score.item(), label.item()))

    precisions = []
    for u in user_scores:
        ranked = sorted(user_scores[u], reverse=True)[:k]
        hits = sum(label for _, label in ranked)
        precisions.append(hits / k)

    return np.mean(precisions)

print("Precision@10:", precision_at_k(model, test_loader))

# Generate Recommendations
def recommend_songs(model, user_idx, top_n=10):
    model.eval()

    user_tensor = torch.tensor([user_idx] * num_songs).to(device)
    song_tensor = torch.arange(num_songs).to(device)

    with torch.no_grad():
        scores = model(user_tensor, song_tensor)

    top_songs = torch.topk(scores, top_n).indices.cpu().numpy()
    return top_songs