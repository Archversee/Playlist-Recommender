🎵 Playlist Recommender System

A hybrid music recommendation system using BPR-Neural Collaborative Filtering (BPR-NCF) and content-based similarity.
The system can recommend tracks to users based on their listening history while also handling cold-start scenarios.

Project Overview:
The Playlist Recommender combines collaborative filtering and content-based filtering:

- Collaborative Filtering (BPR-NCF): Learns user and track embeddings from implicit feedback (listens).
- Content-Based Filtering: Uses track audio features (danceability, energy, tempo, etc.) to find similar tracks.
- Hybrid Recommendation: Weighted combination of CF + CB to improve recommendations and handle cold-start users.

Features:
- Train a BPR-NCF model on user listening history.
- Precompute content similarity matrix for tracks.
- Hybrid recommendations that combine CF and content-based scores.
- Evaluate using Recall@K and NDCG@K metrics.
- Supports saving/loading the trained model to avoid retraining.
- Handles cold-start users using content similarity fallback.

Project Structure:
Playlist-Recommender/
│
├─ data/
│   ├─ music_info.csv           # Track metadata & audio features
│   └─ listening_history.csv    # User listening data
│
├─ datasets/
│   └─ bpr_dataset.py           # BPR Dataset & loss function
│
├─ models/
│   ├─ bpr_ncf.py               # BPR-NCF model
│   └─ recommender.py           # Hybrid Recommender class
│
├─ utils/
│   ├─ data_utils.py            # Data loading & encoding
│   ├─ content_utils.py         # Content similarity functions
│   └─ metrics.py               # Recall & NDCG evaluation
│
├─ train.py                     # Training script for BPR
├─ main.py                      # Main pipeline (train/eval/recommend)
└─ README.md

Usage:
Run the main pipeline
- python main.py

What it does:
1. Loads and encodes user & track data.
2. Computes content similarity between tracks.
3. Splits data into train/test sets.
4. Trains or loads a BPR-NCF model.
5. Evaluates model with Recall@10 and NDCG@10.
6. Outputs top recommendations for a sample user.


Theory & Approach:
1. Implicit Feedback
Only positive interactions (listens > 0) are considered.
Converts listen counts to binary labels (1 = listened, 0 = not listened).

2. BPR-Neural Collaborative Filtering
Learns user & track embeddings.

Uses pairwise ranking loss:
- 𝐿 = −∑(𝑢,𝑖,𝑗)ln𝜎(𝑠𝑢𝑖−𝑠𝑢𝑗)
Encourages the model to rank positive items higher than negatives.

3. Content-Based Filtering
Uses track audio features to calculate similarity between tracks.
Useful for cold-start users or new tracks.

4. Hybrid Recommendation
Combines collaborative filtering and content-based scores:
final_score = 𝛼 ⋅ CF_score + (1−𝛼) ⋅ CB_score
α = 0.7 by default (70% CF, 30% content)

5. Evaluation
Recall@K: fraction of relevant tracks in top-K recommendations.
NDCG@K: considers position of relevant tracks in top-K.

Evaluates only on tracks the user hasn't seen in training.

👤 Sunwei Neo
Built as a learning project for recommender systems using deep learning.
