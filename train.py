# Training Script

import torch
from torch.utils.data import DataLoader
from datasets.bpr_dataset import BPRDataset, bpr_loss
from tqdm import tqdm

def train_bpr(model, train_df, train_pos_dict, num_items,
              epochs=2, batch_size=512, lr=1e-3, device="cpu"):

    print("train_df Full size:", len(train_df))
    train_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print("train_df sub-sampple size:", len(train_df))
    loader = DataLoader(
        BPRDataset(train_df, num_items, train_pos_dict),
        batch_size=batch_size,
        shuffle=True
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for u, p, n in tqdm(loader, desc=f"Epoch {epoch+1}"):
            u, p, n = u.to(device), p.to(device), n.to(device)
            optim.zero_grad()
            loss = bpr_loss(model(u, p), model(u, n))
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
