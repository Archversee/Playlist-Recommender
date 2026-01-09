import pandas as pd
import pickle
import os


def load_data(music_path, listen_path):
    music_df = pd.read_csv(music_path)
    listen_df = pd.read_csv(listen_path)

    listen_df['listen'] = (listen_df['playcount'] > 0).astype(int)
    listen_df = listen_df[['user_id', 'track_id', 'listen']]

    common_tracks = set(music_df['track_id']) & set(listen_df['track_id'])
    music_df = music_df[music_df['track_id'].isin(common_tracks)]
    listen_df = listen_df[listen_df['track_id'].isin(common_tracks)]

    return music_df, listen_df


def encode_users_tracks(
    listen_df,
    music_df,
    user_path="user2idx.pkl",
    track_path="track2idx.pkl"
):
    if os.path.exists(user_path) and os.path.exists(track_path):
        with open(user_path, "rb") as f:
            user2idx = pickle.load(f)
        with open(track_path, "rb") as f:
            track2idx = pickle.load(f)
    else:
        user2idx = {u: i for i, u in enumerate(listen_df.user_id.unique())}
        track2idx = {t: i for i, t in enumerate(music_df.track_id.unique())}

        with open(user_path, "wb") as f:
            pickle.dump(user2idx, f)
        with open(track_path, "wb") as f:
            pickle.dump(track2idx, f)

    listen_df['user_idx'] = listen_df['user_id'].map(user2idx)
    listen_df['track_idx'] = listen_df['track_id'].map(track2idx)
    music_df['track_idx'] = music_df['track_id'].map(track2idx)

    return listen_df, music_df, user2idx, track2idx
