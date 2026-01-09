import torch
import numpy as np
from tqdm import tqdm


def recall_ndcg_at_k(
    model,
    train_pos_dict,
    test_pos_dict,
    num_tracks,
    device,
    k=10,
    max_users=5000
):
    model.eval()
    recalls, ndcgs = [], []

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

            user_tensor = torch.full(
                (num_tracks,),
                user,
                dtype=torch.long,
                device=device
            )
            item_tensor = torch.arange(num_tracks, device=device)

            scores = model(user_tensor, item_tensor)

            if seen_items:
                scores[list(seen_items)] = -1e9

            top_k = torch.topk(scores, min(k, num_tracks)).indices.cpu().numpy()
            hits = np.isin(top_k, list(pos_items)).astype(int)

            recall = hits.sum() / len(pos_items)
            ndcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))

            recalls.append(recall)
            ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)
