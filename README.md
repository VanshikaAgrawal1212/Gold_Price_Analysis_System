# Gold Price Analysis & Prediction

A complete, modular machine-learning project that forecasts gold prices from
macro drivers. It covers the full path: data acquisition → EDA → feature
engineering → model comparison → evaluation → deployment.

```
goldprice analysis system/
├── configs/
│   └── settings.py           # tickers, horizon, lags, hyper-parameters (edit here)
├── src/
│   ├── data_loader.py        # Yahoo Finance + FRED download, CSV cache, synthetic fallback
│   ├── eda.py                # quality report, cleaning, correlations, figures
│   ├── features.py           # lags, rolling stats, calendar, splits, scaling
│   ├── models.py             # baselines, Ridge, Random Forest, XGBoost
│   ├── lstm_model.py         # optional Keras LSTM (guarded import)
│   ├── evaluate.py           # RMSE / MAE / R², hit rate, skill score, plots
│   ├── pipeline.py           # orchestration: py -m src.pipeline
│   └── service.py            # shared serving layer (used by both apps)
├── scripts/
│   ├── run_eda.py            # step 3 on its own, printed to the console
│   └── predict.py            # CLI forecast from the saved bundle
├── apps/
│   ├── streamlit_app.py      # dashboard
│   └── api.py                # FastAPI inference service
├── data/raw/                 # download cache (gitignored)
├── figures/                  # all generated plots (gitignored)
└── artifacts/                # model bundle, metrics CSV, metadata (gitignored)
```

---

## 1. Problem statement & approach

Gold price forecasting is **both** a regression problem and a time-series
problem, and the distinction decides the design:

| | Plain regression | Time-series formulation |
|---|---|---|
| Row independence | assumes i.i.d. rows | violates it — today depends on yesterday |
| Train/test split | random `train_test_split` | **chronological, never random** |
| Target | price level | forward **return** (stationary) |
| Failure mode | leaks the future via shuffling | leaks via centred windows |

**This project uses the time-series formulation of a regression problem.**
Concretely:

1. The *model* is a regressor (Random Forest, XGBoost, LSTM) — so the loss and
   metrics are regression metrics.
2. The *target* is the forward return `gold[t+h] / gold[t] - 1`, not the price
   level. A level model must learn that gold was 1200 in 2011 and 4000 today;
   a return model sees the same stationary distribution across both regimes.
3. All splits are chronological with an **embargo gap** of `h` rows, because a
   training label at `t` secretly contains prices up to `t+h`.
4. `random_split` is never used. A single shuffled split gives an R² near 0.99
   and a completely useless model — it has memorised the price level.

Realistic expectation: at a 5-day horizon, daily-bar gold is close to
efficiently priced. A good model removes 1–5% of the naive baseline's error.
The pipeline reports `skill_vs_naive` precisely so that a marginal result is
visible instead of hidden behind a flattering R².

## 2. Dataset & features

| Column | Yahoo/FRED symbol | Why it is in the model |
|---|---|---|
| `gold_close` (target) | `GC=F` | COMEX front-month futures, USD/oz |
| `usdinr` | `INR=X` | Indian demand is the largest physical bid; converts to ₹/gram |
| `crude_oil` | `CL=F` | inflation expectations / energy-cost channel |
| `sp500` | `^GSPC` | risk appetite — gold and equities swap sign by regime |
| `us10y` | `^TNX` | real-rate proxy, gold's main competitor (non-yielding) |
| `dxy` | `DX-Y.NYB` | gold is priced in dollars; mechanically negative link |
| `silver` | `SI=F` | precious-complex co-movement, gold/silver ratio |
| `gld` | `GLD` | ETF spot proxy, liquidity/positioning signal |
| `cpi`, `cpi_yoy` | FRED `CPIAUCSL` | inflation; combined with `us10y` → real yield |

**Sources.** Yahoo Finance via `yfinance` (free, daily bars, ~40 years of
history) and FRED's keyless CSV endpoint for CPI. Alternatives with cleaner
licensing: `fredapi`, `pandas_datareader`, Alpha Vantage, Stooq
(`stooq.io` CSVs), or NSE/BSE for MCX gold.

**Practical notes**

- `START_DATE` in `configs/settings.py` controls history depth; 2010 gives
  ~4,000 rows — enough for walk-forward CV.
- Every download is cached in `data/raw/*.csv`, so the pipeline re-runs offline
  and results stay reproducible. Delete the folder to force a fresh pull.
- Inflation is *monthly* while prices are *daily*. The YoY change is computed
  at monthly frequency and only then forward-filled onto the daily calendar —
  computing `pct_change(12)` on the daily copy returns 0.0 for most rows.

## 3. Data preprocessing & EDA

```bash
py scripts/run_eda.py
```

Produces console tables and these files in `figures/`:

| Figure | What to look for |
|---|---|
| `00_missing_pattern.png` | isolated white gaps = holidays; vertical bands = broken feed |
| `01_price_trend.png` | gold with 63-day rolling mean/std band, plus rebased overlays |
| `02_correlation_heatmap.png` | Spearman on levels **and** Pearson on returns |
| `02b_target_drivers.png` | ranked correlation with gold — drives feature selection |
| `03_distributions.png` | per-asset return histograms with skew/kurtosis |
| `quality_report.csv`, `adf_test.csv` | missingness, ranges, stationarity p-values |

Cleaning rules implemented in `src/eda.py::clean`:

1. drop columns >40% missing;
2. mask non-positive prices (negative WTI in Apr 2020 breaks `log` and ratios);
3. linear-interpolate gaps ≤5 days, `ffill/bfill` the edges, leave longer runs
   as NaN rather than inventing prices;
4. drop rows with no target label;
5. **flag** moves >25% instead of clipping them — flattening a real crash
   teaches the model that crashes do not happen.

Two diagnostics that matter more than the heatmaps:

* **Spearman on levels is spurious.** Two series both rising since 2010
  correlate at 0.8 for no economic reason. The returns heatmap next to it is
  the honest one.
* **ADF stationarity test** (`p < 0.05`) shows almost every *level* is
  non-stationary while returns are stationary — the empirical justification for
  modelling returns.

## 4. Feature engineering & scaling

`src/features.py` builds ~200 features in five blocks:

```python
add_target_derived_features   # gold returns, 21/63d realised vol, vol regime,
                              # 21/63 MA ratio, change sign
add_returns_features          # 1/5/21d returns + log returns for every driver
add_lag_features              # lagged levels at 1,2,3,5,10,21 days
add_rolling_features          # rolling mean, std, min-max position, z-score
add_macro_spread_features     # gold/silver, gold/oil, real-yield proxy,
                              # dollar pressure, INR-quoted gold
add_calendar_features         # dow/month/quarter + sin-cos encodings,
                              # Indian festive season, month start/end
```

**Leakage rules** (the difference between a demo and a deployable model):

* only `shift(+lag)` — never `shift(-lag)` — on features;
* `rolling(center=False)` only;
* the label `y_target = pct_change(h).shift(-h)` is the *only* place future
  data appears;
* `gold_close` itself is excluded from the feature list;
* scalers and imputers are fitted on the training slice and never re-fitted on
  test data.

**Scaling.** `RobustScaler` (median/IQR) is the default because return
distributions have fat tails — `StandardScaler`'s mean/σ gets dragged by crash
days. Tree models skip scaling entirely (split points are scale-invariant), so
`ForestModel` and `BoostedModel` only impute. `lagged_impute` fills NaN/inf
with training-set medians, keeping the whole transform inside the pipeline
object that gets pickled with the model.

**Splitting.** `time_series_split` produces contiguous train/valid/test blocks
separated by an `h`-row embargo. `walk_forward_folds` provides expanding-window
folds for rolling-origin validation, which is the only scheme that resembles
how the model will actually be used in production (retrain monthly, predict
forward).

## 5. Model building

```bash
py -m src.pipeline                 # baselines + Ridge + RF + XGBoost
py -m src.pipeline --lstm          # add the Keras LSTM
py -m src.pipeline --horizon 10    # 10 business days ahead
py -m src.pipeline --synthetic     # offline code-path test (metrics meaningless)
```

| Model | Role in the comparison |
|---|---|
| `naive_persistence` | ŷ = 0 — the bar every real model must clear |
| `naive_drift` | ŷ = historical mean move |
| `naive_momentum` | OLS on yesterday's return; many "ML" results are this |
| `ridge` | linear benchmark, α chosen by `RidgeCV` over `TimeSeriesSplit` |
| `random_forest` | bagged trees, robust to non-linearity and outliers |
| `xgboost` | gradient boosting with early stopping on the valid block |
| `lstm` | 2×LSTM → Dense, Huber loss, sequence windows of 60 days |

All seven share one `fit`/`predict` contract, so `evaluate.py` treats them
identically and adding a model is one entry in `get_models()`.

Notable implementation details:

* XGBoost uses `early_stopping_rounds` against the *chronological* valid block;
  the wrapper degrades gracefully if no valid set is supplied.
* The LSTM target is `huber` loss, so a handful of panic days do not dominate
  the gradient.
* Ridge needed `alpha` cross-validated: with a return target whose σ ≈ 0.03, a
  fixed `alpha=1.0` is effectively no regularisation and produced R² ≈ −3.5.
* TensorFlow is imported lazily inside functions; without it the pipeline logs
  a warning and continues. It needs its own Python 3.11/3.12 virtualenv
  (`requirements-lstm.txt`) — TF publishes no wheels for very new Pythons.

## 6. Model evaluation

```bash
# artifacts/metrics_test.csv and figures/04-07
```

Metrics, reported in **two spaces**:

| Metric | Return space | Price space (USD/oz) |
|---|---|---|
| RMSE | `rmse_ret` | `rmse_usd` |
| MAE | `mae_ret` | `mae_usd` |
| R² | `r2_ret` | — |
| Hit rate | `directional_accuracy` over *called* days | |
| **Skill vs naive** | `1 - model_rmse_usd / naive_rmse_usd` | |

and four figures:

* `04_predictions.png` — actual vs. each model's implied price path;
* `05_errors.png` — violin + histogram of USD errors (an off-centre mean is a
  bias you can fix; a wide one is risk you cannot);
* `06_walk_forward.png` — per-fold RMSE, which exposes instability a single
  hold-out hides;
* `07_feature_importance.png` — top drivers for the winning model.

**Why `R²` alone is misleading here.** Over the 2024-2026 test window gold made
a large directional move. A model that predicts "flat" gets a terrible R² yet a
competitive RMSE, and a model with R² = 0.02 can still be tradable if its hit
rate is 55%. Report RMSE + skill + hit rate together.

Interpreting the current run on real data (2010-2026, h = 5, ~657 test rows):
XGBoost is the best model but its skill over naive is roughly **+0.5%**, and
walk-forward CV is negative in most folds. That is the honest answer for daily
gold, and the reason the baselines are in the table.

## 7. Deployment

Two options sharing one `src/service.py`, so a forecast means the same thing in
both.

**Streamlit — human-facing dashboard**

```bash
py -m streamlit run apps/streamlit_app.py
```

Four tabs: Forecast (with a ±1 RMSE band and the implied ₹/gram price),
Back-test (metrics table + figures), EDA (interactive price and correlation
heatmaps), Features (the grouped feature dictionary and importances). The
sidebar has a **Retrain models** button.

**FastAPI — machine-facing service**

```bash
py -m uvicorn apps.api:app --port 8000
# http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + loaded model, `503` before training |
| `POST /predict` | body `{"model":"xgboost","refresh_data":false}` |
| `GET /forecast/{h}?model=` | GET-friendly variant |
| `GET /models` | names in the bundle |
| `GET /metrics` | held-out metrics for all models |
| `GET /history?days=250` | recent gold + drivers as JSON |

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d "{\"model\": \"xgboost\"}"
```

**CLI / cron**

```bash
py scripts/predict.py --json
```

**Going further.** Containerise with a `gunicorn -k uvicorn.workers.UvicornWorker`
command and mount `artifacts/` as a volume; schedule `py -m src.pipeline`
weekly (cron / GitHub Actions) so the bundle refreshes; keep the model out of
git and store bundles in S3/GCS with the run id in `model_meta.json`. Add a
`/predict` latency log and a drift check on the incoming feature vector before
scoring — a macro-regime change shows up as shifted features long before it
shows up as worse RMSE.

---

## Quick start

```bash
# 1. install  (Python 3.10+; a 3.11/3.12 venv is safest if you want the LSTM)
py -m pip install -r requirements.txt

# 2. explore the data
py scripts/run_eda.py

# 3. train and compare all models (~2 minutes on CPU)
py -m src.pipeline

# 4. look at figures/ and artifacts/metrics_test.csv
# 5. serve it
py -m streamlit run apps/streamlit_app.py
py -m uvicorn apps.api:app --port 8000
```

## Known limitations

* Daily bars capture daily information; positioning data (COT reports), ETF
  flows, and real-rate breakevens would add signal the current set lacks.
* One fixed horizon. Multi-horizon targets (1/5/21d) or a direct multi-step
  LSTM would be the next modelling step.
* No transaction costs, no bid/ask, no MCX Premium to COMEX for the ₹ figure.
* Results are for education. **Not investment advice.**
