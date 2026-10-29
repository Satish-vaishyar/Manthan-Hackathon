"""Train model3 (Deep Momentum Network) on GPU by maximising the net Sharpe ratio.

  python src/train.py            # n_seeds ensemble
  python src/train.py --smoke    # 3 epochs, 1 seed (pipeline check)

Writes artifacts/seed_<k>/best.pt, history.json, artifacts/metrics.json, artifacts/test_positions.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import backtest
from dataset import SequenceDataset, eval_windows, load
from model import DMN

ROOT = Path(__file__).resolve().parents[1]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def sharpe_loss(pos, r, sigma, tc: dict) -> tuple[torch.Tensor, float]:
    lev = torch.clamp(tc["sigma_target"] / 252 ** 0.5 / sigma.clamp(min=1e-6), max=tc["max_leverage"])
    w = pos * lev
    turnover = torch.cat([torch.zeros_like(w[:, :1]), (w[:, 1:] - w[:, :-1]).abs()], 1)
    ret = (w * r - tc["cost_bps"] * 1e-4 * turnover)[:, tc["burn_in"]:]
    sharpe = ret.mean() / (ret.std() + 1e-9) * 252 ** 0.5
    return -sharpe, float(sharpe)


@torch.no_grad()
def predict(model: DMN, X: np.ndarray, device, bs: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pos, trend = [], []
    for i in range(0, len(X), bs):
        p, t = model(torch.from_numpy(X[i:i + bs]).to(device))
        pos.append(p[:, -1].cpu().numpy()); trend.append(torch.sigmoid(t[:, -1]).cpu().numpy())
    return np.concatenate(pos), np.concatenate(trend)


def to_grid(z: dict, split: int, values: np.ndarray, d_idx, t_idx):
    days = np.where(z["split"] == split)[0]
    grid = np.full((len(days), z["X"].shape[1]), np.nan, dtype=np.float32)
    row = {d: k for k, d in enumerate(days)}
    grid[[row[d] for d in d_idx], t_idx] = values
    return grid, days


def evaluate_split(z: dict, split: int, pos_values, d_idx, t_idx, tc: dict, long_only: bool = False) -> dict:
    grid, days = to_grid(z, split, pos_values, d_idx, t_idx)
    if long_only:
        grid = np.clip(grid, 0, None)
    mask = ~np.isnan(grid)
    return backtest.run(np.nan_to_num(grid), z["r_next"][days], z["sigma"][days], mask,
                        sigma_target=tc["sigma_target"], max_leverage=tc["max_leverage"], cost_bps=tc["cost_bps"])


def train_one(cfg: dict, z: dict, seed: int, device, smoke: bool) -> tuple[Path, dict]:
    set_seed(seed)
    tc, mc = cfg["train"], cfg["model"]
    L = tc["seq_len"]
    Xv, dv, tv = eval_windows(z, 1, L)
    model = DMN(**mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    out_dir = ROOT / cfg["export"]["out_dir"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 3 if smoke else tc["epochs"]
    best, best_ep, bad, hist = -1e9, -1, 0, []
    for ep in range(epochs):
        # New sequence offsets every epoch = more distinct windows without overlap inside an epoch.
        ds = SequenceDataset(z, 0, L, tc["stride"], offset=ep)
        dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True)
        model.train()
        t0, tr_sh = time.time(), []
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            pos, tlog = model(b["x"])
            loss_sh, sh = sharpe_loss(pos, b["r"], b["sigma"], tc)
            m = b["m_trend"].float()
            bce = (F.binary_cross_entropy_with_logits(tlog, b["y_trend"], reduction="none") * m).sum() / m.sum()
            # Sharpe is unchanged if every position is scaled, so anchor the average size.
            expo = (pos[:, tc["burn_in"]:].abs().mean() - tc["exposure_target"]) ** 2
            loss = tc["loss"]["sharpe"] * loss_sh + tc["loss"]["trend"] * bce + tc["loss"]["exposure"] * expo
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            tr_sh.append(sh)
        pv, _ = predict(model, Xv, device)
        vm = evaluate_split(z, 1, pv, dv, tv, tc)
        hist.append({"epoch": ep, "train_sharpe": float(np.mean(tr_sh)) if tr_sh else None,
                     **{f"val_{k}": v for k, v in vm.items()}})
        print(f"[seed {seed}] ep {ep:3d}  train Sharpe {np.mean(tr_sh) if tr_sh else float('nan'):+.2f}  "
              f"val net Sharpe {vm['sharpe']:+.2f}  turnover {vm.get('avg_turnover', 0):.3f}  "
              f"{time.time() - t0:.1f}s ({len(ds)} seqs)", flush=True)
        if vm["sharpe"] > best:
            best, best_ep, bad = vm["sharpe"], ep, 0
            torch.save({"state_dict": model.state_dict(), "model_cfg": mc, "seq_len": L, "epoch": ep,
                        "val": vm, "seed": seed}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"[seed {seed}] early stop at {ep}, best epoch {best_ep}")
                break
    json.dump(hist, open(out_dir / "history.json", "w"), indent=2)
    return out_dir / "best.pt", {"seed": seed, "best_epoch": best_ep, "best_val_sharpe": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model3.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
                                 else "  WARNING: no GPU found, training on CPU"))
    z = load(str(ROOT / cfg["data"]["processed"]))
    tc = cfg["train"]
    seeds = [tc["seed"] + k for k in range(1 if args.smoke else tc["n_seeds"])]
    ckpts, summ = [], []
    for s in seeds:
        ck, sm = train_one(cfg, z, s, device, args.smoke)
        ckpts.append(ck); summ.append(sm)

    res = {"seeds": summ}
    for split, name in ((1, "val"), (2, "test")):
        Xs, ds_, ts_ = eval_windows(z, split, tc["seq_len"])
        members = []
        for ck in ckpts:
            st = torch.load(ck, map_location=device, weights_only=True)
            m = DMN(**st["model_cfg"]).to(device)
            m.load_state_dict(st["state_dict"])
            members.append(predict(m, Xs, device))
        pos = np.mean([p for p, _ in members], axis=0)
        trend = np.mean([t for _, t in members], axis=0)
        res[name] = {"long_short": evaluate_split(z, split, pos, ds_, ts_, tc),
                     "long_only": evaluate_split(z, split, pos, ds_, ts_, tc, long_only=True)}
        if name == "test":
            pd.DataFrame({"date": z["dates"][ds_], "ticker": z["tickers"][ts_], "position": pos,
                          "p_trend_up": trend, "r_next": z["r_next"][ds_, ts_]}).to_csv(
                ROOT / cfg["export"]["out_dir"] / "test_positions.csv", index=False)
    json.dump(res, open(ROOT / cfg["export"]["out_dir"] / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
