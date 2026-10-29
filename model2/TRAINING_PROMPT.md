# model2 — Complete Training Prompt

Copy everything between the two `=====` lines and give it to the person or AI coding assistant on the **GPU machine**.
Part A (recording) must run on a laptop connected to the **venue Wi-Fi**; Parts B–D run on the GPU machine.

=====================================================================================================

## ROLE
You are an ML engineer training **model2**, a GPU-fine-tuned news model for the MANTHAN 2026 autonomous trading
hackathon. Each simulated day the exchange streams daily candles and then (150 ms later) news headlines per ticker
over Kafka. Orders fill at that day's close. model2 decides which headlines are actually about the company and
predicts each ticker's **next-day excess return** from them. Its 8 outputs feed the trading strategy and model1.

## WHAT YOU HAVE (folder `model2/`)
```
model2/
├── configs/model2.yaml            hyper-parameters, paths, Kafka address
├── configs/aliases_manual.json    hand-checked company brand names per ticker
├── data/raw/captures/<run>/       raw Kafka captures (ticks.jsonl, news.jsonl)
├── data/processed/                news.csv, prices.csv, aliases_auto.json, dataset.parquet (generated)
├── src/record_kafka.py            records Kafka ticks+news (venue network only)
├── src/consolidate.py             merges captures, removes duplicates across sandbox loops, mines aliases
├── src/text.py                    cleaning, company masking, alias mining (also used live)
├── src/prepare.py                 labels, price context, FinBERT teacher, weak relevance labels, splits
├── src/dataset.py / model.py      ticker-day dataset; NRIM = FinBERT + relevance-gated pooling + heads
├── src/train.py                   GPU fine-tuning, early stopping, seed ensemble, metrics
├── src/baseline.py                zero-shot FinBERT baselines (the bar to beat)
├── src/export.py / infer.py       packaging + NewsScorer used by the bot (latency check included)
├── ARCHITECTURE.md                read first
└── DATA_REPORT.md                 read first (data problems and how they are handled)
```

## DATA FACTS YOU MUST RESPECT
1. Current data is too small (114 training ticker-days). `train.py` refuses a full run below 300.
   **Record at least one full sandbox loop first (Part A).**
2. Most headlines are unrelated to the company. Never assume a headline is relevant.
3. The model must not learn stock identity: symbols are stripped and aliases become "the company"
   (`prepare.py` prints `ticker symbols left in model text: none`; it must say none).
4. Labels: next-day log return minus the cross-ticker mean, winsorised ±15%, only when the next candle is
   ≤ 5 calendar days later. Splits are by date. **Never tune on the test split.**
5. Everything must run offline at the venue: keep the Hugging Face cache with FinBERT.

## PART A — record data (venue laptop on the competition Wi-Fi, can run while model1 trains)
```bash
cd model2
pip install kafka-python pyyaml
python src/record_kafka.py --minutes 45
```
- 45 minutes ≈ one full 500-day loop at 5 s/day. Record **two loops (90 min) if time allows**.
- It uses its own consumer group, so it does not disturb the team's bot.
- Check the summary it prints: news ≈ 15–20 per simulated day; `loops_seen` should contain one complete loop.
- Then copy the new `data/raw/captures/<run_id>/` folder to the GPU machine's `model2/data/raw/captures/`.

## PART B — environment (GPU machine, from `model2/`)
```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux: source .venv/bin/activate
nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the driver's CUDA version
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True`. The first run of `prepare.py` downloads FinBERT (~440 MB) into the Hugging Face cache.

## PART C — build data, baseline, train
```bash
python src/consolidate.py          # check: unique news count, dates, tickers; review the printed aliases
```
Review `data/processed/aliases_auto.json`. Add real brand names for each ticker (e.g. "DSNY": ["DreamScape"]) to
`configs/aliases_manual.json`. Only CamelCase auto-aliases are used automatically.
```bash
python src/prepare.py              # FinBERT teacher on GPU; must print "ticker symbols left in model text: none"
python src/baseline.py             # zero-shot FinBERT baselines -> artifacts/baseline_metrics.json
python src/train.py --smoke        # 1-epoch pipeline check, then delete artifacts/seed_* and metrics
python src/train.py                # full run: 3 seeds x <= 15 epochs
```
Expected: `prepare.py` reports ≥ 300 train ticker-days; relevant share ≈ 10–30%.

### Optional small search (validation split only, ≤ 12 trials, pick by val `ic_pooled`)
| Knob | Values |
|---|---|
| `model.unfreeze_layers` | 2, 4, 6 |
| `train.lr_encoder` | 1e-5, 2e-5, 3e-5 |
| `train.loss.relevance` | 0.3, 0.5, 1.0 |
| `text.max_items_per_day` | 8, 16 |
Use copies of the config; retrain the chosen one once; only then read the test metrics.

## PART D — acceptance and export
Acceptance (all must hold to ship):
- [ ] ≥ 500 test ticker-days.
- [ ] Test `ic_pooled` > the better of the two zero-shot baselines, and > 0.03.
- [ ] Val and test IC have the same sign.
- [ ] Test `top20_minus_bottom20` > 0.
- [ ] `relevance_auc_vs_weak_labels` > 0.8 on test.

```bash
python src/export.py
```
Check the printout: fp32/int8 CPU latency per member and `int8 max abs diff` (use int8 in the bot only if < 0.05).
If the criteria fail, say so: the bot will use the zero-shot FinBERT + relevance filter baseline instead.

## DELIVERABLES
1. `artifacts/`: `model2_seed*.pt`, `model2_meta.json`, `metrics.json`, `baseline_metrics.json`,
   `test_predictions.csv`, `seed_*/history.json`.
2. The final config and `configs/aliases_manual.json`.
3. `artifacts/REPORT.md`: GPU, data size (ticker-days per split), tried settings with val IC, final val/test vs
   baselines, acceptance checklist, latency numbers.
4. The Hugging Face cache folder for `ProsusAI/finbert`
   (`~/.cache/huggingface/hub/models--ProsusAI--finbert`), so the bot laptop works offline.

## RULES
- Do not put ticker symbols, company names or IDs into model inputs.
- Do not use external news or prices to unmask companies.
- Keep FinBERT's lower layers frozen; the data is small.
- Fix seeds; everything must be reproducible from the config.

=====================================================================================================
