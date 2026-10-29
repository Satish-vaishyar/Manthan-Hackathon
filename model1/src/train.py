"""Train model1 (CSTT) on GPU (falls back to CPU).

  python src/train.py --config configs/model1.yaml                 # full run, n_seeds ensemble
  python src/train.py --config configs/model1.yaml --smoke         # 2 epochs, 1 seed (pipeline check)

Writes artifacts/seed_<k>/best.pt, artifacts/metrics.json, artifacts/test_predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import yaml
from torch.utils.data import DataLoader

from dataset import DayDataset, collate_days
from model import CSTT

ROOT = Path(__file__).resolve().parents[1]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def masked_ic(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean over days of the Pearson correlation between pred and y across valid tickers."""
    m = m.float()
    n = m.sum(1).clamp(min=1)
    pc = pred - (pred * m).sum(1, keepdim=True) / n.unsqueeze(1)
    yc = y - (y * m).sum(1, keepdim=True) / n.unsqueeze(1)
    cov = (pc * yc * m).sum(1)
    den = torch.sqrt((pc.pow(2) * m).sum(1) * (yc.pow(2) * m).sum(1)).clamp(min=1e-8)
    ok = (m.sum(1) >= 3).float()
    return ((cov / den) * ok).sum() / ok.sum().clamp(min=1)


def day_zscore(y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    mf = m.float()
    n = mf.sum(1, keepdim=True).clamp(min=1)
    mu = (y * mf).sum(1, keepdim=True) / n
    sd = torch.sqrt(((y - mu).pow(2) * mf).sum(1, keepdim=True) / n).clamp(min=1e-4)
    return ((y - mu) / sd).clamp(-4, 4) * mf


def compute_loss(out: dict, b: dict, w: dict) -> tuple[torch.Tensor, dict]:
    m1, m5 = b["nmask"], b["nmask"] & b["m5"]
    z1, z5 = day_zscore(b["y1"], m1), day_zscore(b["y5"], m5)
    ic1, ic5 = masked_ic(out["ret1"], b["y1"], m1), masked_ic(out["ret5"], b["y5"], m5)
    hub = (Fn.huber_loss(out["ret1"], z1, reduction="none", delta=1.0) * m1).sum() / m1.sum() \
        + (Fn.huber_loss(out["ret5"], z5, reduction="none", delta=1.0) * m5).sum() / m5.sum().clamp(min=1)
    bce = (Fn.binary_cross_entropy_with_logits(out["dir"], (b["y1"] > 0).float(), reduction="none")
           * m1).sum() / m1.sum()
    vol = (Fn.mse_loss(out["vol"], b["yvol"], reduction="none") * m5).sum() / m5.sum().clamp(min=1)
    mkt = Fn.binary_cross_entropy_with_logits(out["mkt"], b["ymkt"])
    loss = (-w["ic_h1"] * ic1 - w["ic_h5"] * ic5 + w["huber"] * hub + w["direction"] * bce
            + w["vol"] * vol + w["regime"] * mkt)
    return loss, {"ic1": ic1.item(), "ic5": ic5.item(), "huber": hub.item(), "bce": bce.item(),
                  "vol": vol.item(), "mkt": mkt.item()}


@torch.no_grad()
def predict(model: CSTT, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    ds: DayDataset = loader.dataset  # type: ignore[assignment]
    rows = []
    for b in loader:
        bd = {k: v.to(device) for k, v in b.items()}
        out = model(bd["x"], bd["tmask"], bd["nmask"])
        for i, d in enumerate(b["day"].tolist()):
            tick = np.where(ds._eligible(d))[0]
            k = len(tick)
            for name, col in (("ret1", "pred1"), ("ret5", "pred5"), ("dir", "pdir"), ("vol", "pvol")):
                v = out[name][i, :k].float().cpu().numpy()
                if name == "dir":
                    v = 1 / (1 + np.exp(-v))
                rows.append(pd.DataFrame({"day": d, "ticker_idx": tick, col: v}))
    df = pd.concat(rows).groupby(["day", "ticker_idx"], as_index=False).first()
    df["date"] = ds.dates[df["day"]]
    df["ticker"] = ds.tickers[df["ticker_idx"]]
    df["y1"] = ds.y1[df["day"], df["ticker_idx"]]
    df["y5"] = ds.y5[df["day"], df["ticker_idx"]]
    return df


def evaluate(df: pd.DataFrame, pred_col: str = "pred1", y_col: str = "y1", top_k: int = 5) -> dict:
    ics, spreads, hits = [], [], []
    for _, g in df.groupby("date"):
        if len(g) < 3:
            continue
        ics.append(g[pred_col].rank().corr(g[y_col].rank()))
        k = min(top_k, len(g) // 2)
        s = g.sort_values(pred_col)
        spreads.append(s[y_col].tail(k).mean() - s[y_col].head(k).mean())
        hits.append(((g[pred_col] > g[pred_col].median()) == (g[y_col] > 0)).mean())
    ics, spreads = np.array(ics), np.array(spreads)
    return {
        "days": int(len(ics)),
        "ic_mean": float(np.nanmean(ics)), "ic_std": float(np.nanstd(ics)),
        "icir": float(np.nanmean(ics) / (np.nanstd(ics) + 1e-9)),
        "ic_tstat": float(np.nanmean(ics) / (np.nanstd(ics) + 1e-9) * math.sqrt(max(len(ics), 1))),
        "topk_minus_bottomk_daily": float(np.nanmean(spreads)),
        "hit_rate": float(np.nanmean(hits)),
    }


def train_one(cfg: dict, seed: int, device: torch.device, smoke: bool) -> tuple[Path, dict]:
    set_seed(seed)
    dc, mc, tc = cfg["data"], cfg["model"], cfg["train"]
    npz = str(ROOT / dc["processed"])
    aug = {k: tc[k] for k in ("ticker_subsample", "feature_noise", "feature_dropout", "history_dropout")}
    tr = DayDataset(npz, 0, dc["window"], dc["min_history"], train=True, aug=aug, seed=seed)
    va = DayDataset(npz, 1, dc["window"], dc["min_history"])
    kw = dict(collate_fn=collate_days, num_workers=0 if smoke else tc["num_workers"],
              pin_memory=device.type == "cuda")
    tl = DataLoader(tr, batch_size=tc["batch_days"], shuffle=True, drop_last=True, **kw)
    vl = DataLoader(va, batch_size=tc["batch_days"], shuffle=False, **kw)

    model = CSTT(window=dc["window"], **mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = 2 if smoke else tc["epochs"]
    steps = epochs * max(len(tl), 1)
    warm = tc["warmup_epochs"] * max(len(tl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    use_amp = tc["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    out_dir = ROOT / cfg["export"]["out_dir"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_ep, bad, history = -1e9, -1, 0, []
    for ep in range(epochs):
        model.train()
        t0, agg = time.time(), []
        for b in tl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(b["x"], b["tmask"], b["nmask"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = compute_loss(out, b, tc["loss"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            agg.append(parts)
        vm = evaluate(predict(model, vl, device), top_k=cfg["eval"]["top_k"])
        tr_ic = float(np.mean([a["ic1"] for a in agg])) if agg else float("nan")
        history.append({"epoch": ep, "train_ic1": tr_ic, **{f"val_{k}": v for k, v in vm.items()}})
        print(f"[seed {seed}] ep {ep:3d}  train IC {tr_ic:+.4f}  val IC {vm['ic_mean']:+.4f} "
              f"(t={vm['ic_tstat']:+.2f})  spread {vm['topk_minus_bottomk_daily']:+.5f}  "
              f"{time.time() - t0:.1f}s", flush=True)
        if vm["ic_mean"] > best:
            best, best_ep, bad = vm["ic_mean"], ep, 0
            torch.save({"state_dict": model.state_dict(), "model_cfg": mc, "window": dc["window"],
                        "epoch": ep, "val": vm, "seed": seed}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"[seed {seed}] early stop at epoch {ep}, best epoch {best_ep}")
                break
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    return out_dir / "best.pt", {"seed": seed, "best_epoch": best_ep, "best_val_ic": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model1.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
                                 "  WARNING: no GPU found, training on CPU"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    n_seeds = 1 if args.smoke else cfg["train"]["n_seeds"]
    seeds = [cfg["train"]["seed"] + k for k in range(n_seeds)]
    ckpts, summaries = [], []
    for s in seeds:
        ck, summ = train_one(cfg, s, device, args.smoke)
        ckpts.append(ck); summaries.append(summ)

    # Ensemble evaluation on val and test (average of standardised seed predictions).
    dc = cfg["data"]
    npz = str(ROOT / dc["processed"])
    results = {"seeds": summaries}
    for split, name in ((1, "val"), (2, "test")):
        ds = DayDataset(npz, split, dc["window"], dc["min_history"])
        dl = DataLoader(ds, batch_size=cfg["train"]["batch_days"], collate_fn=collate_days)
        preds = []
        for ck in ckpts:
            st = torch.load(ck, map_location=device, weights_only=True)
            m = CSTT(window=st["window"], **st["model_cfg"]).to(device)
            m.load_state_dict(st["state_dict"])
            p = predict(m, dl, device)
            for c in ("pred1", "pred5"):
                p[c] = p.groupby("date")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
            preds.append(p)
        ens = preds[0].copy()
        for c in ("pred1", "pred5", "pdir", "pvol"):
            ens[c] = np.mean([p[c].values for p in preds], axis=0)
        results[name] = {"h1": evaluate(ens, "pred1", "y1", cfg["eval"]["top_k"]),
                         "h5": evaluate(ens, "pred5", "y5", cfg["eval"]["top_k"])}
        if name == "test":
            ens.drop(columns=["day", "ticker_idx"]).to_csv(ROOT / cfg["export"]["out_dir"] / "test_predictions.csv",
                                                           index=False)
    out = ROOT / cfg["export"]["out_dir"] / "metrics.json"
    json.dump(results, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "seeds"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
