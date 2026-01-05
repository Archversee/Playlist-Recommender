import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os
import pickle

MODEL_PATH = "bpr_ncf_model.pth"

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
similarity_matrix = cosine_similarity(content_matrix).astype(np.float32)
print("Content similarity matrix:", similarity_matrix.shape)

# Build track_idx → content row mapping
track_idx_to_content_row = {
    tid: idx for idx, tid in enumerate(content_df.track_idx.values)
}

# =========================
# 4. Train / Test Split
# =========================
interactions_train, interactions_test = train_test_split(
    listen_df[['user_idx', 'track_idx', 'listen']],
    test_size=0.2,
    random_state=42
)

user_pos_dict = (
    listen_df[listen_df.listen == 1]
    .groupby('user_idx')['track_idx']
    .apply(set)
    .to_dict()
)

train_pos_dict = (
    interactions_train[interactions_train.listen == 1]
    .groupby('user_idx')['track_idx']
    .apply(set)
    .to_dict()
)

test_pos_dict = (
    interactions_test[interactions_test.listen == 1]
    .groupby('user_idx')['track_idx']
    .apply(set)
    .to_dict()
)

class BPRDataset(Dataset):
    def __init__(self, df, num_items, user_pos_dict):
        self.users = df['user_idx'].values
        self.pos_items = df['track_idx'].values
        self.num_items = num_items
        self.user_pos_dict = user_pos_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        # Faster negative sampling
        user_pos = self.user_pos_dict[user]
        neg_item = np.random.randint(0, self.num_items)
        while neg_item in user_pos:
            neg_item = np.random.randint(0, self.num_items)
        
        return torch.tensor(user), torch.tensor(pos_item), torch.tensor(neg_item)

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

if os.getenv("FULL_TRAIN", "0") == "0":
    # Use 10% of training data for quick testing
    interactions_train = interactions_train.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Using subsampled train set: {len(interactions_train)} interactions")

# =========================
# 5. BPR 
# =========================
class BPRNCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BPRNCF(num_users, num_tracks).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 6. Train or Load Model BPR
# =========================
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loaded trained BPR model from disk.")
else:
    print("Training BPR model from scratch...")
    positive_interactions = interactions_train[interactions_train.listen == 1]

    train_loader = DataLoader(
        BPRDataset(positive_interactions, num_tracks, train_pos_dict),
        batch_size=512,
        shuffle=True,
    )

    model = BPRNCF(num_users, num_tracks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # Add progress bar
        for users, pos_items, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} done. Loss: {total_loss:.4f}. Time: {epoch_time/60:.2f} min")
    
    torch.save(model.state_dict(), MODEL_PATH)

# =========================
# 7. Ranking Evaluation (Recall@K, NDCG@K)
# =========================
def recall_ndcg_at_k(model, train_pos_dict, test_pos_dict, k=10, max_users=5000):
    model.eval()
    recalls, ndcgs = [], []
    
    # Sample users to avoid days-long eval
    test_users = list(test_pos_dict.keys())
    if len(test_users) > max_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, size=max_users, replace=False)

    with torch.no_grad():
        for user in tqdm(test_users, desc="Evaluating"):
            pos_items = test_pos_dict[user]
            seen_items = train_pos_dict.get(user, set())
            if len(pos_items) == 0:
                continue

            # Batch scoring: all items at once
            user_tensor = torch.full((num_tracks,), user, dtype=torch.long, device=device)
            item_tensor = torch.arange(num_tracks, device=device)
            scores = model(user_tensor, item_tensor)

            # Mask seen items
            if seen_items:
                scores[list(seen_items)] = -1e9

            top_k = torch.topk(scores, min(k, num_tracks)).indices.cpu().numpy()
            hits = np.isin(top_k, list(pos_items)).astype(int)

            recall = hits.sum() / len(pos_items)
            ndcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))

            recalls.append(recall)
            ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)

# Evaluate model 
recall, ndcg = recall_ndcg_at_k(model, train_pos_dict, test_pos_dict)
print(f"Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")

# =========================
# 8. Generate Recommendations
# =========================
class Recommender:
    def __init__(self, model, similarity_matrix, track_idx_to_content_row,
                 train_pos_dict=None, alpha=0.7, device=None):
        """
        Args:
            model: trained BPR-NCF model
            similarity_matrix: precomputed content similarity matrix
            track_idx_to_content_row: dict mapping track_idx -> row in similarity matrix
            train_pos_dict: dict of user -> set of tracks already seen (for masking)
            alpha: float, weight for CF vs content
            device: torch device
        """
        self.model = model
        self.similarity_matrix = similarity_matrix
        self.track_idx_to_content_row = track_idx_to_content_row
        self.train_pos_dict = train_pos_dict or {}
        self.alpha = alpha
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tracks = len(track_idx_to_content_row)

    def _content_only(self, track_idx, top_n=10):
        """Content-based recommendation for cold-start"""
        if track_idx not in self.track_idx_to_content_row:
            return np.random.choice(list(self.track_idx_to_content_row.keys()),
                                    size=top_n, replace=False)

        seed_row = self.track_idx_to_content_row[track_idx]
        scores = self.similarity_matrix[seed_row]

        # Map scores back to track_idx order
        all_track_indices = np.array(list(self.track_idx_to_content_row.keys()))
        scores = scores[list(self.track_idx_to_content_row.values())]

        # Exclude seed track itself
        seed_pos = np.where(all_track_indices == track_idx)[0][0]
        scores[seed_pos] = -1e9

        top_indices = np.argsort(scores)[-top_n:][::-1]
        return all_track_indices[top_indices]

    def recommend(self, user_idx, top_n=10):
        """Hybrid recommendation"""
        # Cold-start fallback
        if user_idx not in self.train_pos_dict or len(self.train_pos_dict[user_idx]) == 0:
            # Pick a random track the user hasn't listened to as seed
            seed_track = np.random.choice(list(self.track_idx_to_content_row.keys()))
            return self._content_only(seed_track, top_n)

        self.model.eval()

        # Compute CF scores
        user_tensor = torch.tensor([user_idx] * self.num_tracks).to(self.device)
        item_tensor = torch.arange(self.num_tracks).to(self.device)
        with torch.no_grad():
            cf_scores = self.model(user_tensor, item_tensor).cpu().numpy()

        # Mask already seen tracks
        listened = self.train_pos_dict.get(user_idx, set())
        for t in listened:
            cf_scores[t] = -1e9

        # Compute content scores
        content_scores = np.zeros(self.num_tracks)
        listened_rows = [
            self.track_idx_to_content_row[t]
            for t in listened
            if t in self.track_idx_to_content_row
        ]

        if listened_rows:
            valid_tracks = list(self.track_idx_to_content_row.keys())
            rows = [self.track_idx_to_content_row[t] for t in valid_tracks]
            # max similarity to any listened track
            content_scores[valid_tracks] = self.similarity_matrix[rows][:, listened_rows].max(axis=1)
            for t in listened:
                content_scores[t] = -1e9

        # Normalize scores
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)

        # Hybrid score
        final_scores = self.alpha * cf_scores + (1 - self.alpha) * content_scores

        top_items = np.argsort(final_scores)[-top_n:][::-1]
        return top_items




# top_tracks_idx = recommend_songs(model, user_idx=20, top_n=10)
# print("Top recommended songs for user 20:")
# print(music_df[music_df['track_idx'].isin(top_tracks_idx)][['track_id','name','artist']])

# for u in [0, 20, 500]:
#     recs = recommend_songs(model, u, user_pos_dict)
#     print(f"\nUser {u} recommendations:")
#     print(music_df[music_df.track_idx.isin(recs)][['name','artist']])


recommender = Recommender(
    model=model,
    similarity_matrix=similarity_matrix,
    track_idx_to_content_row=track_idx_to_content_row,
    train_pos_dict=train_pos_dict,
    alpha=0.7,
    device=device
)

# Get top-10 recommendations for user 20
recs = recommender.recommend(user_idx=20, top_n=10)
print(music_df[music_df.track_idx.isin(recs)][['name','artist']])