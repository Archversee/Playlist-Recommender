import os
import torch
from sklearn.model_selection import train_test_split

from utils.data_utils import load_data, encode_users_tracks
from utils.content_utils import build_content_similarity
from utils.metrics import recall_ndcg_at_k
from models.bpr_ncf import BPRNCF
from models.recommender import Recommender
from train import train_bpr

MODEL_PATH = "bpr_ncf_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. Load Data
# =========================
music_df, listen_df = load_data(
    "data/music_info.csv",
    "data/listening_history.csv"
)

listen_df, music_df, user2idx, track2idx = encode_users_tracks(
    listen_df, music_df
)

# =========================
# 2. Content Similarity
# =========================
sim_matrix, track_map = build_content_similarity(music_df)

# =========================
# 3. Train / Test Split
# =========================
positive_df = listen_df[listen_df.listen == 1]
train_df, test_df = train_test_split(
    positive_df,
    test_size=0.2,
    random_state=42
)

train_pos_dict = (
    train_df.groupby('user_idx')['track_idx']
    .apply(set)
    .to_dict()
)

test_pos_dict = (
    test_df.groupby('user_idx')['track_idx']
    .apply(set)
    .to_dict()
)

# =========================
# 4. Model
# =========================
model = BPRNCF(len(user2idx), len(track2idx)).to(device)

# Load existing model if available
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Loaded trained BPR model from disk.")
else:
    # Train the model
    train_bpr(
        model=model,
        train_df=train_df,
        train_pos_dict=train_pos_dict,
        num_items=len(track2idx),
        device=device,
        epochs=2
    )
    # Save trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model trained and saved to disk.")

# =========================
# 5. Evaluate
# =========================
recall, ndcg = recall_ndcg_at_k(
    model=model,
    train_pos_dict=train_pos_dict,
    test_pos_dict=test_pos_dict,
    num_tracks=len(track2idx),
    device=device,
    k=10
)

print(f"\nRecall@10: {recall:.4f}")
print(f"NDCG@10:  {ndcg:.4f}")

# =========================
# 6. Recommend
# =========================
recommender = Recommender(
    model=model,
    similarity_matrix=sim_matrix,
    track_idx_to_content_row=track_map,
    train_pos_dict=train_pos_dict,
    alpha=0.7,
    device=device
)

recs = recommender.recommend(user_idx=4, top_n=10)
print("\nTop recommendations for user 4:")
print(music_df[music_df.track_idx.isin(recs)][['name', 'artist']])
