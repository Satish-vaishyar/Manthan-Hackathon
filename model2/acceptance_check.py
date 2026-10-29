"""Read existing artifacts/ and report acceptance status. Does NOT re-train anything."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

base = json.load(open(ART / "baseline_metrics.json"))
metr = json.load(open(ART / "metrics.json"))
meta = json.load(open(ART / "model2_meta.json"))

test = metr["test"]
val = metr["val"]
base_test = base["test"]
base_bar = max(base_test["zero_shot_all"]["ic_pooled"],
               base_test["zero_shot_relevant"]["ic_pooled"])
base_bar_name = ("zero_shot_all" if base_test["zero_shot_all"]["ic_pooled"] >= base_test["zero_shot_relevant"]["ic_pooled"]
                 else "zero_shot_relevant")

checks = {
    "test_n >= 500":                (test["n"] >= 500, f"n={test['n']}"),
    f"test_ic_pooled > {base_bar_name} ({base_bar:+.3f})":
                                    (test["ic_pooled"] > base_bar, f"{test['ic_pooled']:+.4f} vs bar {base_bar:+.4f}"),
    "test_ic_pooled > 0.03":        (test["ic_pooled"] > 0.03, f"{test['ic_pooled']:+.4f}"),
    "val/test IC same sign":        ((val["ic_pooled"] * test["ic_pooled"]) > 0,
                                     f"val={val['ic_pooled']:+.3f}, test={test['ic_pooled']:+.3f}"),
    "top20_minus_bottom20 > 0":     (test["top20_minus_bottom20"] > 0,
                                     f"{test['top20_minus_bottom20']:+.5f}"),
    "relevance_auc > 0.8":          (test.get("relevance_auc_vs_weak_labels", 0) > 0.8,
                                     f"{test.get('relevance_auc_vs_weak_labels', float('nan')):.3f}"),
}

print("=" * 70)
print(f"TEST SET: n={test['n']}, dates={test.get('days_ic', '?')}, "
      f"IC pooled={test['ic_pooled']:+.4f}, top20-bot20={test['top20_minus_bottom20']:+.5f}")
print(f"VAL   SET: n={val['n']}, IC pooled={val['ic_pooled']:+.4f}")
print(f"BASELINE: best test = {base_bar_name} = {base_bar:+.4f}")
print(f"CPU latency (per day, {meta.get('cpu_ms_day_items')} items): "
      f"fp32={meta.get('cpu_ms_day_fp32')} ms, int8={meta.get('cpu_ms_day_int8')} ms")
print(f"int8 max abs diff: {meta.get('int8_max_abs_diff')} (use int8 only if < 0.05)")
print("=" * 70)
all_pass = True
for k, (ok, info) in checks.items():
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}]  {k:50s}  ({info})")
    all_pass = all_pass and ok
print("=" * 70)
print("ACCEPTANCE:", "ALL PASSED" if all_pass else "FAILED -- bot should use zero-shot baseline")
print("=" * 70)