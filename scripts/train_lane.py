import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models_def.tiny_unet import TinyUNet
from utils.dataset_helpers import LaneDataset

def main(args):
    ds = LaneDataset(args.data, size=256)
    n = len(ds)
    v = int(n*0.15)
    t = n - v
    train_ds, val_ds = random_split(ds, [t, v])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = TinyUNet()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device!="cpu" else "cpu")
    model.to(device)

    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    best = 999

    for ep in range(args.epochs):
        model.train()
        tl = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(train_loader)

        # validation
        model.eval()
        vl = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out,y)
                vl += loss.item()
        vl /= len(val_loader)

        print(f"Epoch {ep+1}/{args.epochs} - Train {tl:.4f}  Val {vl:.4f}")

        if vl < best:
            best = vl
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_lane_unet.pt")
            print("✔ Saved best model!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/lanes_synthetic")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    main(args)
