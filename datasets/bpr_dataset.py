#Dataset & loss

import torch
import numpy as np
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, df, num_items, user_pos_dict):
        self.users = df.user_idx.values
        self.items = df.track_idx.values
        self.num_items = num_items
        self.user_pos_dict = user_pos_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.items[idx]

        neg = np.random.randint(self.num_items)
        while neg in self.user_pos_dict[u]:
            neg = np.random.randint(self.num_items)

        return (
            torch.tensor(u),
            torch.tensor(pos),
            torch.tensor(neg),
        )


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
