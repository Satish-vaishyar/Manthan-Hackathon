"""Fine-tune model2 (NRIM) on GPU.

  python src/train.py            # full run (n_seeds), refuses if there is too little labelled news
  python src/train.py --smoke    # 1 epoch, 1 seed, small subset (pipeline check)

Writes artifacts/seed_<k>/best.pt (trainable weights only), history.json, artifacts/metrics.json,
artifacts/test_predictions.csv.
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
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import TickerDayDataset
from model import NRIM

ROOT = Path(__file__).resolve().parents[1]


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def batch_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8)


def compute_loss(o: dict, b: dict, w: dict) -> tuple[torch.Tensor, dict]:
    m = b["item_mask"].float()
    ret = F.huber_loss(o["ret"], b["y_ret"].clamp(-4, 4))
    ic = batch_corr(o["ret"], b["y_ret"]) if len(o["ret"]) > 2 else o["ret"].sum() * 0
    dirl = F.binary_cross_entropy_with_logits(o["dir"], (b["y_ret"] > 0).float())
    absl = F.mse_loss(o["abs"], b["y_abs"])
    rel = (F.binary_cross_entropy_with_logits(o["rel"], b["relevant"], reduction="none") * m).sum() / m.sum()
    tm = m * b["has_teacher"].float().unsqueeze(-1)
    kl = (F.kl_div(F.log_softmax(o["sent"], -1), b["teacher"], reduction="none").sum(-1) * tm).sum() \
        / tm.sum().clamp(min=1)
    loss = (w["ret"] * ret - w["ic"] * ic + w["direction"] * dirl + w["absmove"] * absl
            + w["relevance"] * rel + w["distill"] * kl)
    return loss, {"ret": ret.item(), "ic": ic.item(), "dir": dirl.item(), "rel": rel.item(), "kl": kl.item()}


@torch.no_grad()
def predict(model: NRIM, loader: DataLoader, device, amp_dtype, use_amp) -> pd.DataFrame:
    model.eval()
    ds = loader.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    rows = []
    for b in loader:
        bd = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            o = model(bd["ids"], bd["att"], bd["item_mask"], bd["ctx"])
        rel = torch.sigmoid(o["rel"].float()) * bd["item_mask"]
        for j, i in enumerate(b["idx"].tolist()):
            date, ticker = base.keys(i)
            k = int(b["item_mask"][j].sum())
            rows.append({"date": date, "ticker": ticker, "pred": float(o["ret"][j]),
                         "p_up": float(torch.sigmoid(o["dir"][j].float())), "abs_pred": float(o["abs"][j]),
                         "rel_max": float(rel[j].max()), "n_items": k, "y_ret": float(b["y_raw"][j]),
                         "rel_pred": rel[j, :k].cpu().numpy().tolist(),
                         "rel_true": b["relevant"][j, :k].numpy().tolist()})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, col: str = "pred") -> dict:
    if len(df) < 3:
        return {"n": len(df), "ic_pooled": float("nan")}
    ic_pooled = df[col].rank().corr(df["y_ret"].rank())
    daily = [g[col].rank().corr(g["y_ret"].rank()) for _, g in df.groupby("date") if len(g) >= 3]
    q = df[col].rank(pct=True)
    spread = df.loc[q > 0.8, "y_ret"].mean() - df.loc[q <= 0.2, "y_ret"].mean()
    res = {"n": int(len(df)), "ic_pooled": float(ic_pooled),
           "ic_daily_mean": float(np.nanmean(daily)) if daily else float("nan"), "days_ic": len(daily),
           "top20_minus_bottom20": float(spread)}
    if "p_up" in df and df["y_ret"].gt(0).nunique() == 2:
        res["dir_auc"] = float(roc_auc_score(df["y_ret"] > 0, df["p_up"]))
    if "rel_pred" in df:
        rp = np.concatenate(df["rel_pred"].map(np.array).to_list())
        rt = np.concatenate(df["rel_true"].map(np.array).to_list())
        if len(np.unique(rt)) == 2:
            res["relevance_auc_vs_weak_labels"] = float(roc_auc_score(rt, rp))
    return res


def loaders(cfg: dict, df: pd.DataFrame, tok, seed: int, smoke: bool):
    tc, xc = cfg["train"], cfg["text"]
    y_std = float(df.loc[df["split"] == 0, "y_ret"].std() or 0.01)
    sets = {s: TickerDayDataset(df[df["split"] == s], tok, xc["max_items_per_day"], xc["max_len"], y_std,
                                train=(s == 0), seed=seed) for s in (0, 1, 2)}
    if smoke:
        sets = {s: Subset(d, list(range(min(len(d), 48)))) for s, d in sets.items()}
    bs = tc["batch_ticker_days"]
    return (DataLoader(sets[0], batch_size=bs, shuffle=True, drop_last=len(sets[0]) > bs),
            DataLoader(sets[1], batch_size=bs), DataLoader(sets[2], batch_size=bs), y_std)


def train_one(cfg, df, tok, seed, device, smoke) -> tuple[Path, dict]:
    set_seed(seed)
    tc, mc = cfg["train"], cfg["model"]
    tl, vl, _, y_std = loaders(cfg, df, tok, seed, smoke)
    model = NRIM(**mc).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder_params(), "lr": tc["lr_encoder"]},
                             {"params": model.head_params(), "lr": tc["lr_head"]}], weight_decay=tc["weight_decay"])
    epochs = 1 if smoke else tc["epochs"]
    steps = max(epochs * len(tl), 1)
    sched = get_cosine_schedule_with_warmup(opt, int(tc["warmup_ratio"] * steps), steps)
    use_amp = tc["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    out_dir = ROOT / cfg["export"]["out_dir"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_ep, bad, hist = -1e9, -1, 0, []
    for ep in range(epochs):
        model.train()
        t0, parts = time.time(), []
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                o = model(b["ids"], b["att"], b["item_mask"], b["ctx"])
            o = {k: v.float() for k, v in o.items()}
            loss, p = compute_loss(o, b, tc["loss"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            parts.append(p)
        vm = evaluate(predict(model, vl, device, amp_dtype, use_amp))
        score = vm.get("ic_pooled", float("nan"))
        hist.append({"epoch": ep, **{f"train_{k}": float(np.mean([p[k] for p in parts])) for k in parts[0]},
                     **{f"val_{k}": v for k, v in vm.items()}} if parts else {"epoch": ep})
        print(f"[seed {seed}] ep {ep:2d}  train IC {np.mean([p['ic'] for p in parts]) if parts else float('nan'):+.3f}"
              f"  val IC pooled {score:+.3f}  val rel AUC {vm.get('relevance_auc_vs_weak_labels', float('nan')):.3f}"
              f"  {time.time() - t0:.0f}s", flush=True)
        if not math.isnan(score) and score > best or best_ep < 0:
            best, best_ep, bad = (score if not math.isnan(score) else -1e9), ep, 0
            # keep every non-encoder tensor plus the fine-tuned encoder layers
            n_layers = model.enc.config.num_hidden_layers
            keep_layers = {f"enc.encoder.layer.{i}." for i in range(n_layers - mc["unfreeze_layers"], n_layers)}
            trainable = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("enc.") or any(k.startswith(p) for p in keep_layers)}
            torch.save({"state_dict": trainable, "model_cfg": mc, "y_std": y_std, "epoch": ep, "val": vm,
                        "seed": seed}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"[seed {seed}] early stop at {ep}, best epoch {best_ep}")
                break
    json.dump(hist, open(out_dir / "history.json", "w"), indent=2)
    return out_dir / "best.pt", {"seed": seed, "best_epoch": best_ep, "best_val_ic_pooled": best}


def load_model(ck: Path, device) -> tuple[NRIM, dict]:
    st = torch.load(ck, map_location="cpu", weights_only=True)
    m = NRIM(**st["model_cfg"])
    m.load_state_dict(st["state_dict"], strict=False)
    return m.to(device).eval(), st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "model2.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
                                 else "  WARNING: no GPU found, training on CPU"))
    df = pd.read_parquet(ROOT / cfg["data"]["dataset"])
    df = df[df["split"] >= 0]
    n_td = df[df["split"] == 0].groupby(["date", "ticker"]).ngroups
    print(f"labelled ticker-days: train {n_td}, val {df[df['split'] == 1].groupby(['date', 'ticker']).ngroups}, "
          f"test {df[df['split'] == 2].groupby(['date', 'ticker']).ngroups}")
    if not args.smoke and n_td < cfg["train"]["min_train_ticker_days"]:
        raise SystemExit(f"Only {n_td} training ticker-days (< {cfg['train']['min_train_ticker_days']}). "
                         "Record at least one full sandbox loop first (src/record_kafka.py), then re-run "
                         "consolidate.py and prepare.py.")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    seeds = [cfg["train"]["seed"] + k for k in range(1 if args.smoke else cfg["train"]["n_seeds"])]
    ckpts, summ = [], []
    for s in seeds:
        ck, sm = train_one(cfg, df, tok, s, device, args.smoke)
        ckpts.append(ck); summ.append(sm)

    _, vl, te, _ = loaders(cfg, df, tok, 0, args.smoke)
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    res = {"seeds": summ}
    for name, dl in (("val", vl), ("test", te)):
        preds = []
        for ck in ckpts:
            m, _ = load_model(ck, device)
            p = predict(m, dl, device, amp_dtype, use_amp)
            p["pred"] = (p["pred"] - p["pred"].mean()) / (p["pred"].std() + 1e-9)
            preds.append(p)
            del m
        ens = preds[0].copy()
        for c in ("pred", "p_up", "abs_pred", "rel_max"):
            ens[c] = np.mean([p[c].values for p in preds], axis=0)
        res[name] = evaluate(ens)
        if name == "test":
            ens.drop(columns=["rel_pred", "rel_true"]).to_csv(ROOT / cfg["export"]["out_dir"] / "test_predictions.csv",
                                                              index=False)
    json.dump(res, open(ROOT / cfg["export"]["out_dir"] / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
