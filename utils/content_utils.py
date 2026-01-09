import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

AUDIO_FEATURES = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
]


def build_content_similarity(music_df):
    """
    Builds cosine similarity matrix over audio features.

    Returns:
        similarity_matrix: np.ndarray [num_tracks, num_tracks]
        track_idx_to_content_row: dict {track_idx -> row index}
    """
    content_df = music_df[['track_idx'] + AUDIO_FEATURES].dropna()

    scaler = StandardScaler()
    content_df[AUDIO_FEATURES] = scaler.fit_transform(
        content_df[AUDIO_FEATURES]
    )

    content_matrix = content_df[AUDIO_FEATURES].values
    similarity_matrix = cosine_similarity(content_matrix).astype(np.float32)

    track_idx_to_content_row = {
        tid: i for i, tid in enumerate(content_df.track_idx.values)
    }

    return similarity_matrix, track_idx_to_content_row
