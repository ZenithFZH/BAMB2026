"""Train a chunked blind clone on a (camera-free) LeRobot dataset.

This is the tutorial's BlindClone with the mini-project's "upgrade #2" applied:
instead of one action per frame, the policy predicts a chunk of the next
CHUNK_SIZE actions from the current joint angles, which stops per-frame errors
from compounding. Trains in minutes on CPU.

Usage:
    uv run python train_blind_chunked.py --repo-id bambschool/so101_blind_smoketest \
        --epochs 200 --out blind_chunked.pt
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
CHUNK_SIZE = 20  # actions predicted per query, ~0.7 s at 30 fps


class BlindChunkedClone(nn.Module):
    """MLP from joint angles to a chunk of future actions. No vision."""

    def __init__(self, d_in, d_out, chunk_size, stats, hidden=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_out = d_out
        for name, val in stats.items():
            self.register_buffer(name, val)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out * chunk_size),
        )

    def forward(self, s):
        out = self.net((s - self.x_mean) / self.x_std)
        out = out.view(-1, self.chunk_size, self.d_out)
        return out * self.y_std + self.y_mean


def load_chunked_arrays(repo_id, chunk_size):
    """Return (states, action_chunks) as tensors, chunked within episodes."""
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states = np.stack(hf["observation.state"])
    actions = np.stack(hf["action"])
    ep_idx = np.array(hf["episode_index"])

    X, Y = [], []
    for ep in np.unique(ep_idx):
        s, a = states[ep_idx == ep], actions[ep_idx == ep]
        for t in range(len(s) - chunk_size):
            X.append(s[t])
            Y.append(a[t : t + chunk_size])
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out", default="blind_chunked.pt")
    args = p.parse_args()

    torch.manual_seed(SEED)
    X, Y = load_chunked_arrays(args.repo_id, CHUNK_SIZE)
    print(
        f"{len(X)} training pairs: "
        f"state {tuple(X.shape[1:])} -> chunk {tuple(Y.shape[1:])}"
    )

    stats = {
        "x_mean": X.mean(0), "x_std": X.std(0).clamp(min=1e-4),
        "y_mean": Y.mean((0, 1)), "y_std": Y.std((0, 1)).clamp(min=1e-4),
    }
    model = BlindChunkedClone(X.shape[1], Y.shape[2], CHUNK_SIZE, stats)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss {epoch_loss / len(loader.dataset):.5f}")

    # Save the action ranges seen in training: deployment clamps to these.
    action_low = Y.reshape(-1, Y.shape[2]).min(0).values
    action_high = Y.reshape(-1, Y.shape[2]).max(0).values
    torch.save(
        {
            "model": model.state_dict(),
            "d_in": X.shape[1], "d_out": Y.shape[2], "chunk_size": CHUNK_SIZE,
            "action_low": action_low, "action_high": action_high,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
