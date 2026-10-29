# model2 — Analysis Notes (2026-10-29, post-training)

## What model2 is
GPU fine-tune of FinBERT that, for every (ticker, day), decides which headlines are actually about
that company and predicts the **next-day excess return** from them. 8 output features per ticker-day
feed model1 and the strategy. See `ARCHITECTURE.md` and `TRAINING_PROMPT.md` for the full spec.

## Code map (everything checked)

| File | Role |
|------|------|
| `src/record_kafka.py` | Records market.ticks + market.news from the venue Kafka (Part A). Uses its own `recorder-<uuid>` group so it never steals partitions from the bot's `group-<team>`. |
| `src/consolidate.py` | Merges every capture in `data/raw/captures/`, de-dups prices on (date, ticker) and news on (date, ticker, headline), runs `mine_aliases()` on the news. |
| `src/text.py` | `clean()`, `mask()` (own aliases → "the company"; other tickers removed), `mine_aliases()` (CamelCase brand heuristic), `load_aliases()` (manual + auto-CamelCase). |
| `src/prepare.py` | Joins news with per-ticker price context, builds labels (next-day log-return minus cross-ticker mean, winsorised ±15%, only if next candle ≤5 days later), date-ordered 70/15/15 split with 2-day embargo, **runs FinBERT as a teacher** to get `t_pos/t_neg/t_neu`, saves `dataset.parquet`. |
| `src/dataset.py` | `TickerDayDataset`: groups by (date, ticker), tokenises up to K=16 items, pads to T=96 tokens, applies the relevance-prioritised sampler in train mode. |
| `src/model.py` | `NRIM`: per-item FinBERT (top 4 layers unfrozen) → 128-d projection → relevance head + 3-way sentiment head. Ticker-day pooling: relevance-gated softmax attention, then concat [pooled ‖ ctx-MLP(5 candle features) ‖ log(1+Σσ(rel))] → MLP → 3 heads (ret, dir, abs). |
| `src/train.py` | Loss = 1.0·Huber(ret) − 0.5·corr(ret, y) + 0.3·BCE(dir) + 0.2·MSE(abs) + 0.5·BCE(rel, weak) + 0.3·KL(sent, teacher). AdamW with separate LRs (encoder 2e-5, heads 1e-3), cosine schedule with 10% warm-up, AMP bf16/fp16, batch 16 ticker-days, 3-seed ensemble, early stop on val pooled IC. **Refuses a full run below `min_train_ticker_days=300`**. |
| `src/baseline.py` | Two zero-shot FinBERT bars: mean over all items vs. only relevant items. |
| `src/export.py` | Copies each `seed_<k>/best.pt` to `model2_seed<k>.pt`, writes `model2_meta.json`, runs a CPU latency test on a real day for both fp32 and int8 dynamic quantisation. |
| `src/infer.py` | `NewsScorer`: pads to the longest item of the day (not max_len), averages across ensemble members, returns the 8 features. |

## Pipeline outcome

### Final test set
| Metric | Value |
|--------|-------|
| n | 443 |
| unique dates | 72 |
| unique tickers | 10 |
| weak-relevant share | 8.0 % |

### Trained ensemble (3 seeds)
| Metric | Val | Test |
|--------|-----|------|
| IC pooled | +0.097 | **+0.106** |
| top20-bot20 | +0.0097 | +0.0031 |
| direction AUC | 0.506 | 0.511 |
| rel-AUC vs weak labels | 0.695 | 0.552 |

### Zero-shot FinBERT baselines (the bar to beat)
| Metric | Val | Test |
|--------|-----|------|
| zero_shot_all IC pooled | −0.037 | +0.009 |
| zero_shot_relevant IC pooled | −0.052 | **+0.019** ← bar |

### Acceptance: 5/6 pass
| Gate | Status |
|------|--------|
| test_n >= 500 | **FAIL** (443 vs 500) |
| test_ic > baseline | **PASS** (+0.106 vs +0.019, 5.6×) |
| test_ic > 0.03 | **PASS** (3.5×) |
| val/test IC same sign | **PASS** |
| top20_minus_bottom20 > 0 | **PASS** (+0.0031) |
| rel_auc_vs_weak_labels > 0.8 | **FAIL** (0.552) |

### Verdict
The fine-tuned ensemble **clearly beats the baseline** (5.6× the IC bar) but fails two
acceptance gates: test n < 500 (close, recoverable with one more recording) and
rel-AUC < 0.8 (alias weak labels are noisy on test). Per the prompt's rule that **all**
gates must hold to ship, the bot should use the **zero-shot FinBERT + relevance filter
baseline** instead. The trained artifacts are saved for inspection and for any future
retrain once more data / a hand-labelled relevance set is available.

## Lessons learned / recommendations

1. **Alias list curation matters more than volume.** Many CamelCase tokens auto-mined by
   `mine_aliases()` are real-world competitor / context names (`BigBear.ai`, `OneMedNet`,
   `TikTok`, `OpenAI`, `FlixNet`, `OmniSearch`, `DeepSeek`, `Magnus Carlsen`, `Chess`).
   Adding them as aliases would teach the relevance head to mark competitor mentions as
   relevant — a serious false-positive source. The hand-curated `aliases_manual.json`
   only contains true company brands (9 tickers × 1-2 aliases).

2. **The relevance head is the weakest part of the ensemble.** Train rel-AUC saturates
   at 0.99 within 3 epochs, val stays at 0.69, test drops to 0.55. This pattern
   (saturation + generalisation gap) is classic weak-label overfitting. The fix is to
   raise `train.loss.relevance` from 0.5 to ~1.5-2.0 and / or unfreeze fewer encoder
   layers.

3. **The price signal is real and useful.** Test IC pooled +0.106 vs baseline +0.019
   means the model's return prediction is meaningfully better than just averaging
   FinBERT sentiment. This is the headline result and survives the relevance-head
   failure because the price-aware context (`ctx_*` features) carries much of the signal.

4. **Latency is fine.** ~3.3 s/day fp32, ~2 s/day int8 — well under any per-day decision
   deadline. The int8 quantisation increases error too much (max abs diff 0.27 vs 0.05
   threshold) so the bot uses fp32.

5. **Two 45-min recordings = ~3000 labelled ticker-days.** One loop = 500 days × 10
   tickers = 5000 ticker-days raw, ~4500 after the 5-day `max_gap_days` filter and
   deduplication. This is more than enough to train.

## Risk register (resolved)

| # | Risk | Resolution |
|---|------|------------|
| R1 | Recording might not complete a full loop in 45 min | Resolved: each loop took ~50-55 min, so 45 min captured ~80% of a loop |
| R2 | FinBERT teacher OOM on 8 GB GPU | Resolved: `prepare.py` batches 64 at a time, fit fine on RTX 4060 |
| R3 | Test IC not above baseline | Resolved: +0.106 vs +0.019, 5.6× the bar |
| R4 | int8 degradation | Confirmed: 0.27 max abs diff, bot uses fp32 |
| R5 | PowerShell wraps TF stderr as exit-code-1 | Cosmetic — Python returncode is 0 |