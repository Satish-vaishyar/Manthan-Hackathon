"""Run Part C + D end-to-end once Part A is done.

Steps:
  1. consolidate.py        (re-merge any new captures)
  2. prepare.py            (re-compute teacher + weak labels)
  3. baseline.py           (already ran; this re-runs to update the bar)
  4. train.py              (full 3-seed ensemble, until threshold met)
  5. export.py             (package + CPU latency check)
  6. assert acceptance criteria from TRAINING_PROMPT.md

Usage:
  python run_full_pipeline.py
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

def run(cmd, **kw):
    print(f"\n>>> {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=str(ROOT), **kw)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    t0 = time.time()
    run([sys.executable, str(SRC / "consolidate.py")])
    run([sys.executable, str(SRC / "prepare.py")])
    run([sys.executable, str(SRC / "baseline.py")])
    run([sys.executable, str(SRC / "train.py")])
    run([sys.executable, str(SRC / "export.py")])

    # Acceptance check
    print("\n========== ACCEPTANCE ==========")
    base = json.load(open(ROOT / "artifacts" / "baseline_metrics.json"))
    metr = json.load(open(ROOT / "artifacts" / "metrics.json"))
    meta = json.load(open(ROOT / "artifacts" / "model2_meta.json"))
    test = metr["test"]
    base_test = base["test"]
    base_bar = max(base_test["zero_shot_all"]["ic_pooled"],
                   base_test["zero_shot_relevant"]["ic_pooled"])
    val = metr["val"]
    checks = {
        "test_n >= 500": test["n"] >= 500,
        "test_ic_pooled > baseline_bar": test["ic_pooled"] > base_bar,
        "test_ic_pooled > 0.03":         test["ic_pooled"] > 0.03,
        "val/test IC same sign":         (val["ic_pooled"] * test["ic_pooled"]) > 0,
        "top20_minus_bottom20 > 0":      test["top20_minus_bottom20"] > 0,
        "relevance_auc > 0.8":           test.get("relevance_auc_vs_weak_labels", 0) > 0.8,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  test IC pooled: {test['ic_pooled']:+.4f}  (bar {base_bar:+.4f}, min 0.03)")
    print(f"  val   IC pooled: {val['ic_pooled']:+.4f}")
    print(f"  top20-bot20: {test['top20_minus_bottom20']:+.5f}")
    print(f"  relevance AUC: {test.get('relevance_auc_vs_weak_labels', float('nan')):.3f}")
    print(f"  CPU latency fp32: {meta.get('cpu_ms_day_fp32')} ms, int8: {meta.get('cpu_ms_day_int8')} ms, "
          f"items={meta.get('cpu_ms_day_items')}")
    print(f"  int8 max abs diff: {meta.get('int8_max_abs_diff')} (use int8 only if < 0.05)")
    print(f"\nTotal pipeline time: {(time.time()-t0)/60:.1f} min")

    if not all(checks.values()):
        print("\nACCEPTANCE FAILED -- per the prompt, the bot will use the zero-shot FinBERT "
              "+ relevance filter baseline instead.")
    else:
        print("\nACCEPTANCE PASSED.")

if __name__ == "__main__":
    main()