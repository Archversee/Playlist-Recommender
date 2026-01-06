🎵 Music Recommendation System (Neural Collaborative Filtering)

This project implements a music recommendation system using Neural Collaborative Filtering (NCF) with PyTorch. It learns user–song interaction patterns from listening history and recommends songs users are likely to enjoy.
The project also prepares content-based audio similarity features, enabling future hybrid recommendations.

📌 Features
Neural Collaborative Filtering (NCF) with embeddings
Implicit feedback modeling (listened / not listened)
Content-based audio similarity using cosine similarity
Precision@K evaluation
Model persistence (save/load trained models)
GPU support (CUDA if available)

🧠 Model Overview
1. Collaborative Filtering
Users and songs are represented as learned embeddings
A multi-layer perceptron (MLP) predicts the probability that a user will listen to a song

2. Content-Based Similarity
Audio features (danceability, energy, tempo, etc.) are normalized
Cosine similarity measures similarity between songs
⚠️ Currently computed but not yet integrated into final recommendations


📂 Project Structure
.
├── data/
│   ├── music_info.csv
│   └── listening_history.csv
├── user2idx.pkl
├── track2idx.pkl
├── ncf_model.pth
├── app.py   # main script
└── README.md

📊 Datasets
1. music_info.csv : Contains song metadata and audio features:
    1.track_id
    2.name
    3.artist
    4.danceability, energy, tempo, valence, etc.

2. listening_history.csv : Contains user listening behavior:
    1.user_id
    2.track_id
    3.playcount
Listening behavior is converted to binary implicit feedback:
listen = 1 if playcount > 0 else 0


⚙️ Installation
1. pip install torch pandas numpy scikit-learn

(Optional: install CUDA-enabled PyTorch for GPU support)

▶️ Running the Project
python app.py               # Fast (10% data)
FULL_TRAIN=1 python app.py  # Full train (slow)

    What happens:
        1.Data is loaded and cleaned
        2.Users and tracks are encoded
        3.Audio features are scaled and compared
        4.The NCF model is trained (or loaded if already trained)
        5.Precision@10 is evaluated
        6.Top-N song recommendations are generated

📈 Evaluation Metric
Precision@K
Measures how many of the top-K recommended songs were actually listened to:

Precision@K = Relevant songs in top K / K
	​


💾 Model Persistence

Trained model saved as: ncf_model.pth
User and track encodings saved as:
    1. user2idx.pkl
    2. track2idx.pkl

The model will automatically load these files if they exist.

🚀 Future Improvements
Integrate content-based similarity into final recommendations
Add negative sampling for better training
Use ranking loss (e.g., BPR loss)
Add Recall@K / NDCG@K
Cold-start handling for new users or songs
Convert to a full hybrid recommender system

📚 Technologies Used
Python
PyTorch
Pandas & NumPy
Scikit-learn
Cosine Similarity
Neural Collaborative Filtering (NCF)

👤 Sunwei Neo
Built as a learning project for recommender systems using deep learning.
