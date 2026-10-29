"""Export the model3 seed ensemble for CPU inference in the bot.

  python src/export.py

Writes artifacts/model3_seed<k>.onnx and .pt, artifacts/model3_meta.json, and checks CPU latency.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml

from features import FEATURE_NAMES
from model import DMN

ROOT = Path(__file__).resolve().parents[1]


class LastStep(torch.nn.Module):
    def __init__(self, m: DMN):
        super().__init__()
        self.m = m

    def forward(self, x):
        pos, trend = self.m(x)
        return pos[:, -1], torch.sigmoid(trend[:, -1])


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "model3.yaml"))
    art = ROOT / cfg["export"]["out_dir"]
    members = []
    for ck in sorted(art.glob("seed_*/best.pt")):
        st = torch.load(ck, map_location="cpu", weights_only=True)
        m = DMN(**st["model_cfg"]).eval()
        m.load_state_dict(st["state_dict"])
        w = LastStep(m).eval()
        L, F = st["seq_len"], st["model_cfg"]["n_features"]
        name = f"model3_seed{st['seed']}"
        torch.save({"state_dict": m.state_dict(), "model_cfg": st["model_cfg"], "seq_len": L}, art / f"{name}.pt")
        x = torch.randn(10, L, F)
        try:
            torch.onnx.export(w, (x,), art / f"{name}.onnx", opset_version=17, input_names=["x"],
                              output_names=["position", "p_trend_up"],
                              dynamic_axes={"x": {0: "n"}, "position": {0: "n"}, "p_trend_up": {0: "n"}})
            onnx_ok = True
        except Exception as e:
            print(f"ONNX export failed for {name}: {e}")
            onnx_ok = False
        x40 = torch.randn(40, L, F)
        with torch.no_grad():
            w(x40)
            t0 = time.perf_counter()
            for _ in range(20):
                w(x40)
        ms = (time.perf_counter() - t0) / 20 * 1000
        members.append({"seed": st["seed"], "pt": f"{name}.pt", "onnx": f"{name}.onnx" if onnx_ok else None,
                        "val": st["val"], "cpu_ms_40_tickers": round(ms, 2)})
        print(f"{name}: CPU latency for 40 tickers = {ms:.2f} ms")
    if not members:
        raise SystemExit("no artifacts/seed_*/best.pt found — run src/train.py first")
    tc = cfg["train"]
    meta = {"model": "model3-DMN", "seq_len": tc["seq_len"], "features": FEATURE_NAMES,
            "vol_span": cfg["data"]["vol_span"], "sigma_target": tc["sigma_target"],
            "max_leverage": tc["max_leverage"], "members": members,
            "outputs": {"position": "trend-following position signal in [-1, 1] (average the members)",
                        "p_trend_up": "P(next 21-day return > 0)",
                        "target_weight": "position * min(sigma_target/sqrt(252)/sigma_t, max_leverage)",
                        "momentum_regime": "mean position across the live universe (market trend gauge)"}}
    json.dump(meta, open(art / "model3_meta.json", "w"), indent=2)
    print(f"wrote {art / 'model3_meta.json'} with {len(members)} members")


if __name__ == "__main__":
    main()
