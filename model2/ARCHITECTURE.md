# model2 — News Relevance & Impact Model (NRIM)

## 1. Role in the system

model1 reads prices. model2 reads the **news that arrives 150 ms after each candle** and answers, per ticker:
*is any of today's news really about this company, and what will it do to tomorrow's relative return?*
Orders fill at today's close, so news on day D is tradable before its effect on day D+1. This is the main edge
described in `docs/04_STRATEGY.md`.

Output per ticker-day (8 values = model1's `news_dim`):

| # | Feature | Meaning |
|---|---|---|
| 0 | `news_score` | tanh of predicted standardised next-day excess return (−1 … +1) |
| 1 | `p_up` | P(excess return tomorrow > 0) |
| 2 | `abs_move` | expected \|next-day return\| (news-driven volatility) |
| 3 | `rel_max` | highest relevance of today's items |
| 4 | `log1p_n_relevant` | log(1 + expected number of relevant items) |
| 5, 6 | `sent_pos`, `sent_neg` | relevance-weighted sentiment |
| 7 | `has_news` | 1 if the ticker had any news today |

## 2. Network

```
item k of ticker i on day D
  text = clean(headline + ". " + description[:300])
       → own symbol + aliases replaced by "the company"; every other ticker symbol removed
  → FinBERT (ProsusAI/finbert, BERT-base 12 layers; bottom 8 frozen, top 4 fine-tuned)
  → [CLS] 768 → Dropout → Linear(768→128) → GELU = e_k
       ├─ relevance head  Linear(128→1)  → r_k   (weak label: text mentions the company)
       └─ sentiment head  Linear(128→3)  → s_k   (distilled from frozen FinBERT)

ticker-day pooling (up to 16 items)
  a_k = Linear(128→1)(e_k) + log σ(r_k)          relevance-gated attention: irrelevant items get ~0 weight
  v   = Σ softmax(a)_k · e_k
  c   = GELU(Linear(5→32)(day-D candle context: ret1, gap, intraday, hl_range, clv))
  z   = Dropout(GELU(Linear(128+32+1 → 128)([v ‖ c ‖ log(1+Σσ(r_k))])))
  heads: ret = Linear(z) (standardised excess return), dir = Linear(z) (logit), abs = Linear(z) (log|r|)
```
Trainable: top 4 BERT layers (~28M) plus heads (~0.1M). The other ~82M FinBERT parameters stay frozen, which
limits overfitting on a few thousand ticker-days.

## 3. No stock identity

- Every ticker symbol is removed from the text, and the company's own aliases become "the company".
- No ticker embedding, ID or price level is an input. The candle context is scale-free.
- So the model learns *what kind of news moves a stock*, which transfers to whatever tickers trade live.
  `prepare.py` asserts that no symbol is left in the model text.

## 4. Loss

```
L = 1.0·Huber(ret, y_ret/σ)  − 0.5·corr_batch(ret, y_ret)  + 0.3·BCE(dir, y_ret>0)  + 0.2·MSE(abs, log|r|)
  + 0.5·BCE(r_k, mentions_company_k)  + 0.3·KL(FinBERT_teacher_k ‖ softmax(s_k))
```
- `y_ret` is the next-day log return minus the mean over **all** tickers that day, winsorised at ±15%, and only
  labelled when the next candle is ≤ 5 days away (no loop jumps).
- The relevance and distillation terms keep the encoder sensible while return labels are scarce.

## 5. Training setup

| Item | Value |
|---|---|
| Optimiser | AdamW: encoder lr 2e-5, heads lr 1e-3, weight decay 0.01 |
| Schedule | cosine with 10% warm-up |
| Epochs | ≤ 15, early stop (patience 4) on validation pooled IC |
| Batch | 16 ticker-days (≤ 256 news items) |
| Precision | AMP bf16/fp16 on GPU |
| Augmentation | relevant items always kept; other items randomly sampled when there are more than 16 |
| Ensemble | 3 seeds, standardised and averaged |
| Split | by simulated date: 70 / 15 / 15 with a 2-day embargo |

## 6. Evaluation and ship rule

Metrics (`metrics.json`): pooled Spearman IC, mean daily IC, top-20% − bottom-20% next-day excess return,
direction AUC, and relevance AUC against the weak labels.

Ship only if **test pooled IC > the better zero-shot FinBERT baseline** (`baseline_metrics.json`) and > 0.03 with
at least 500 test ticker-days. Otherwise the bot uses zero-shot FinBERT with the relevance filter as its news signal.

## 7. Serving (`src/infer.py`)

`NewsScorer(artifacts).score_day(news_items, candles, tickers, aliases)` → `{ticker: [8 floats]}`.
- Loads FinBERT from the local Hugging Face cache plus the fine-tuned layers from `model2_seed*.pt`.
- Pads to the longest item of the day and uses int8 dynamic quantisation on CPU (≈ 360 ms per member for ~17 items).
- Live aliases: names from `/stocks` plus `mine_aliases()` run on the news received so far.
- Run it in a worker separate from the model1 path; if it misses the decision deadline, the bot uses
  yesterday's decayed news memory.
