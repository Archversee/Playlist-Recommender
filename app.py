import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os
import pickle

# =========================
# 1. Load Data
# =========================
music_df = pd.read_csv("data/music_info.csv")
listen_df = pd.read_csv("data/listening_history.csv")

print("Music DF:", music_df.shape)
print("Listen DF:", listen_df.shape)

listen_df['listen'] = (listen_df['playcount'] > 0).astype(int)
listen_df = listen_df[['user_id', 'track_id', 'listen']]

# Keep only songs that exist in both files
common_tracks = set(music_df['track_id']) & set(listen_df['track_id'])
music_df = music_df[music_df['track_id'].isin(common_tracks)]
listen_df = listen_df[listen_df['track_id'].isin(common_tracks)]

# =========================
# 2. Encode users and tracks
# =========================
if os.path.exists("user2idx.pkl") and os.path.exists("track2idx.pkl"):
    with open("user2idx.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open("track2idx.pkl", "rb") as f:
        track2idx = pickle.load(f)
    print("Loaded user2idx and track2idx from disk.")
else:
    user2idx = {u: i for i, u in enumerate(listen_df['user_id'].unique())}
    track2idx = {t: i for i, t in enumerate(music_df['track_id'].unique())}
    with open("user2idx.pkl", "wb") as f:
        pickle.dump(user2idx, f)
    with open("track2idx.pkl", "wb") as f:
        pickle.dump(track2idx, f)
    print("Created and saved user2idx and track2idx.")

listen_df['user_idx'] = listen_df['user_id'].map(user2idx)
listen_df['track_idx'] = listen_df['track_id'].map(track2idx)
music_df['track_idx'] = music_df['track_id'].map(track2idx)

num_users = len(user2idx)
num_tracks = len(track2idx)
print("Num users:", num_users, "Num tracks:", num_tracks)

# =========================
# 3. Content-based similarity
# =========================
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

content_df = music_df[['track_idx'] + audio_features].dropna()
scaler = StandardScaler()
content_df[audio_features] = scaler.fit_transform(content_df[audio_features])

content_matrix = content_df[audio_features].values
similarity_matrix = cosine_similarity(content_matrix)
print("Content similarity matrix:", similarity_matrix.shape)

# =========================
# 4. Train / Test Split
# =========================
train_df, test_df = train_test_split(
    listen_df[['user_idx', 'track_idx', 'listen']],
    test_size=0.2,
    random_state=42
)

class NCFDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.tracks = torch.tensor(df['track_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(df['listen'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.users[idx], self.tracks[idx], self.labels[idx]

train_loader = DataLoader(NCFDataset(train_df), batch_size=512, shuffle=True)
test_loader = DataLoader(NCFDataset(test_df), batch_size=512)

# =========================
# 5. Neural Collaborative Filtering
# =========================
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_songs, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.song_emb = nn.Embedding(num_songs, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 128),
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralCF(num_users, num_tracks).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 6. Train or Load Model
# =========================
if os.path.exists("ncf_model.pth"):
    model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
    model.eval()
    print("Loaded trained NCF model from disk.")
else:
    print("Training NCF model...")
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
    torch.save(model.state_dict(), "ncf_model.pth")
    print("Training done. Model saved to ncf_model.pth")

# =========================
# 7. Precision@K
# =========================
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
        ranked = sorted(user_scores[u], key=lambda x: x[0], reverse=True)[:k]
        hits = sum(label for _, label in ranked)
        precisions.append(hits / k)
    return np.mean(precisions)

print("Precision@10:", precision_at_k(model, test_loader))

# =========================
# 8. Generate Recommendations
# =========================
def recommend_songs(model, user_idx, top_n=10):
    model.eval()
    user_tensor = torch.tensor([user_idx] * num_tracks).to(device)
    song_tensor = torch.arange(num_tracks).to(device)
    with torch.no_grad():
        scores = model(user_tensor, song_tensor)
    top_songs = torch.topk(scores, top_n).indices.cpu().numpy()
    return top_songs

top_tracks_idx = recommend_songs(model, user_idx=0, top_n=10)
print("Top recommended songs for user 0:")
print(music_df[music_df['track_idx'].isin(top_tracks_idx)][['track_id','name','artist']])
