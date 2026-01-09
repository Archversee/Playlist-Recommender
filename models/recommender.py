import torch
import numpy as np

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
