"""Export the trained seed ensemble for CPU inference inside the bot.

  python src/export.py --config configs/model1.yaml

Writes artifacts/model1_seed<k>.onnx (+ artifacts/model1_seed<k>.pt fallback), artifacts/model1_meta.json
with feature order, scaler, window and ensemble members, and checks CPU latency for a 40-ticker day.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from model import CSTT

ROOT = Path(__file__).resolve().parents[1]


class InferenceWrapper(torch.nn.Module):
    def __init__(self, m: CSTT):
        super().__init__()
        self.m = m

    def forward(self, x, tmask, nmask):
        o = self.m(x, tmask, nmask)
        return o["ret1"], o["ret5"], torch.sigmoid(o["dir"]), o["vol"], torch.sigmoid(o["mkt"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "model1.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    art = ROOT / cfg["export"]["out_dir"]
    members = []
    for ck in sorted(art.glob("seed_*/best.pt")):
        st = torch.load(ck, map_location="cpu", weights_only=True)
        m = CSTT(window=st["window"], **st["model_cfg"]).eval()
        m.load_state_dict(st["state_dict"])
        w = InferenceWrapper(m).eval()
        L, F = st["window"], st["model_cfg"]["n_features"]
        x, tm, nm = torch.randn(1, 10, L, F), torch.ones(1, 10, L, dtype=torch.bool), torch.ones(1, 10, dtype=torch.bool)
        name = f"model1_seed{st['seed']}"
        torch.save({"state_dict": m.state_dict(), "model_cfg": st["model_cfg"], "window": L}, art / f"{name}.pt")
        try:
            torch.onnx.export(w, (x, tm, nm), art / f"{name}.onnx", opset_version=17,
                              input_names=["x", "tmask", "nmask"],
                              output_names=["ret1", "ret5", "p_up", "vol", "p_mkt_up"],
                              dynamic_axes={"x": {1: "n"}, "tmask": {1: "n"}, "nmask": {1: "n"},
                                            "ret1": {1: "n"}, "ret5": {1: "n"}, "p_up": {1: "n"}, "vol": {1: "n"}})
            onnx_ok = True
        except Exception as e:  # ONNX is optional; the .pt file is always written
            print(f"ONNX export failed for {name}: {e}")
            onnx_ok = False
        x40, tm40, nm40 = torch.randn(1, 40, L, F), torch.ones(1, 40, L, dtype=torch.bool), torch.ones(1, 40, dtype=torch.bool)
        with torch.no_grad():
            w(x40, tm40, nm40)
            t0 = time.perf_counter()
            for _ in range(20):
                w(x40, tm40, nm40)
        ms = (time.perf_counter() - t0) / 20 * 1000
        members.append({"seed": st["seed"], "pt": f"{name}.pt", "onnx": f"{name}.onnx" if onnx_ok else None,
                        "val": st["val"], "cpu_ms_40_tickers": round(ms, 2)})
        print(f"{name}: CPU latency for 40 tickers = {ms:.1f} ms")
    scaler = json.load(open(ROOT / Path(cfg["data"]["processed"]).parent / "scaler.json"))
    meta = {"model": "model1-CSTT", "window": cfg["data"]["window"], "min_history": cfg["data"]["min_history"],
            "features": scaler["features"], "scaler": scaler, "members": members,
            "outputs": {"ret1": "relative next-day score (higher = better), standardise per day",
                        "ret5": "relative 5-day score", "p_up": "P(excess return > 0 tomorrow)",
                        "vol": "log mean |r| next 5 days", "p_mkt_up": "P(market up over next 5 days)"}}
    json.dump(meta, open(art / "model1_meta.json", "w"), indent=2)
    print(f"wrote {art / 'model1_meta.json'} with {len(members)} ensemble members")


if __name__ == "__main__":
    main()
