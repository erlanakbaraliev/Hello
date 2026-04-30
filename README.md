# Urban Air Quality Predictor

A Streamlit web application for analysing and forecasting urban air quality (PM2.5 and related pollutants) using ARIMA, LSTM, and XGBoost models.

---

## Entry point

Use the modular multipage app as the canonical runtime from the **`thesis/`** directory:

```bash
cd thesis
streamlit run app/Home.py
```

Entry scripts load [`app/invoke_bootstrap.py`](app/invoke_bootstrap.py) before importing `app` or `database`, so **you do not need `PYTHONPATH`** for normal Streamlit runs. For other tools (tests, REPL imports), you may still use:

```bash
cd thesis
PYTHONPATH=. python -c "import database"
```

On Windows (Command Prompt) for the optional pattern: `set PYTHONPATH=.`.

`app.py` at the repository root is a compatibility launcher only; prefer `streamlit run app/Home.py` above.

---

## Project structure

```
thesis/
├── app.py                  # Compatibility launcher (redirects to modular app)
├── app/
│   ├── Home.py             # Modular entry point (Streamlit multi-page)
│   ├── invoke_bootstrap.py # Prepends thesis/ to sys.path (importlib from entry scripts)
│   ├── logging_utils.py    # Shared logging setup for Streamlit scripts
│   ├── database.py         # Re-exports from root database.py
│   ├── forecasting.py      # Cached model loaders + hourly forecasts (168 h)
│   ├── ui.py               # Shared Streamlit shell helpers + theme injection
│   ├── static/
│   │   ├── theme.css       # Global light-theme styles (loaded by ui.apply_theme)
│   │   └── favicon.png     # Page icon when present
│   └── pages/
│       ├── 1_Data_Explorer.py
│       ├── 2_Predict_future_quality.py
│       ├── 3_Dashboard.py
│       ├── 4_History.py
│       ├── 5_Settings.py
│       └── 6_HelpDesk.py
├── data_processing.py      # Data loading, schema coercion, preprocessing pipeline
├── database.py             # SQLite: users, settings, history, chat
├── models.py               # Daily 7-day forecasts + metadata helpers
├── helpdesk_gemini.py      # Google Gemini chat client
├── tests/                  # Pytest smoke tests for core modules
├── pyproject.toml          # Ruff, Black, pytest, mypy settings
├── environment-appendix.txt # Frozen `pip freeze` for thesis reproducibility (regenerate locally)
├── data/
│   ├── london_2024.csv     # Primary dataset (hourly, full year)
│   └── other_datas/        # LAQN weather data used by merge_london_weather.py
├── models/                 # Saved model artifacts (not committed to git)
│   ├── arima_model.pkl
│   ├── arima_metadata.json
│   ├── lstm_model.keras
│   ├── lstm_scaler.pkl
│   ├── lstm_metadata.json
│   ├── xgboost_model.joblib
│   ├── xgboost_metadata.json
│   └── hashes.json         # Optional SHA-256 manifest for artifact checks
└── notebooks/
    ├── 1_arima_model.ipynb
    ├── 2_lstm_model.ipynb
    └── 3_xgboost_model.ipynb
```

### Streamlit pages (what each screen is for)

| Page | Purpose |
|------|---------|
| **Home** | Sign-in / registration and short scientific context for the thesis demo. |
| **Data Explorer** | Upload any numeric CSV to inspect correlations, distributions, and basic summaries. |
| **Predict future quality** | Upload hourly training-schema CSV; run **168-hour** forecasts with saved notebook models (ARIMA / LSTM / XGBoost). |
| **Dashboard** | Quick view of default London series: latest PM2.5, simple risk band, trends. |
| **History** | List, download, or delete saved prediction runs stored in SQLite. |
| **Settings** | Profile, default model, chart preferences, notifications, password, account deletion. |
| **HelpDesk** | Optional Gemini-powered Q&A; requires API key in secrets or env. |

---

## Forecasting modes (important for the thesis write-up)

The codebase supports **two related but different** forecasting paths. Describe them separately in your methods chapter so examiners see a clear story.

1. **Hourly path (primary app workflow)** — [`app/forecasting.py`](app/forecasting.py)  
   - Used by **Predict future quality** after [`preprocess_upload`](app/forecasting.py) (hourly index, required pollutant columns).  
   - Horizon: **168 hours** (one week of hourly steps).  
   - Loads **saved** notebook artifacts (`arima_model.pkl`, `lstm_model.keras` + scaler, `xgboost_model.joblib`).  
   - Matches the training notebooks’ hourly panel design.

2. **Daily-style path (library-style helpers)** — [`models.py`](models.py) + [`data_processing.py`](data_processing.py)  
   - [`validate_and_prepare`](data_processing.py) resamples to **daily** means; [`predict_next_7_days`](models.py) produces **7 daily** steps (ARIMA refit-on-series, or LSTM/XGBoost on prepared frame).  
   - Useful for experiments and for metadata helpers like [`get_model_metadata`](models.py).  
   - **Not** the same code path as the Streamlit “Predict future quality” page.

**Takeaway for your thesis:** state which path your results and figures come from. If you compare models, say whether the comparison is hourly or daily and that horizons differ (168 h vs 7 days).

---

## Setup

**Project root:** This README lives in the repository root. Open a terminal **in this folder** (next to `requirements.txt` and `app/`). Create **one** virtual environment here as `./.venv`. Avoid a second `.venv` in a parent directory (for example `../.venv`), or Python may load the wrong site-packages when you run Streamlit or tests.

```bash
# Create and activate a virtual environment (inside this repo only)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API keys

Create `.streamlit/secrets.toml` in this same directory (repository root):

```toml
GEMINI_API_KEY = "your-gemini-key"
```

The key is optional — the HelpDesk page degrades gracefully if it is absent.

---

## Running the app

```bash
# From the thesis/ directory:
streamlit run app/Home.py
```

---

## Model artifacts

Trained model files are **not committed to git** (they can be large). Train them by running the notebooks in order:

1. `notebooks/1_arima_model.ipynb` → saves `models/arima_model.pkl`
2. `notebooks/2_lstm_model.ipynb` → saves `models/lstm_model.keras` + `models/lstm_scaler.pkl`
3. `notebooks/3_xgboost_model.ipynb` → saves `models/xgboost_model.joblib`

Each notebook also saves a `*_metadata.json` file with training date, dataset version, and evaluation metrics.

---

## Model metadata

After training, each model has a companion JSON file:

```json
{
  "model": "arima",
  "trained_at": "2025-04-29T12:00:00+00:00",
  "data_version": "london_2024.csv",
  "metrics": { "rmse": 3.21, "mae": 2.45 },
  "notes": "ARIMA(2,1,2) fit on daily PM2.5"
}
```

You can read metadata programmatically:

```python
from models import get_model_metadata
print(get_model_metadata("arima"))
```

---

## Thesis writing: metrics, splits, and what to cite

Keep the **written thesis** aligned with what the code actually does:

| Source | What to report |
|--------|----------------|
| **Notebooks** | Use the printed or saved **RMSE / MAE** (or other metrics) from the notebook evaluation cells, plus train/test window definitions shown there. |
| **`models/*_metadata.json`** | Cite `trained_at`, `data_version`, and the `metrics` dict as the official snapshot of each saved artifact. |
| **Streamlit hourly page** | Figures from **Predict future quality** are **rolling 168-hour** forecasts from saved models, not necessarily the same numeric values as a one-off notebook cell unless you re-run with identical input CSV. |
| **`models.py` daily path** | If you discuss `predict_next_7_days` or `compare_models`, state explicitly that inputs are **daily-resampled** and the horizon is **7 days**. |

**Fair comparison tip:** examiners often ask whether models were compared on the same horizon and preprocessing. If you compare ARIMA vs LSTM, say whether both are hourly (notebook + app path) or daily (`models.py`), and match the text to the figure.

---

## Reproducibility (submission snapshot)

- **Pinned environment for markers:** regenerate after `pip install -r requirements.txt` on the machine you use for final runs, then commit or attach the file with your submission:

  ```bash
  pip freeze > environment-appendix.txt
  ```

  A reference snapshot is already stored as [`environment-appendix.txt`](environment-appendix.txt) (replace it with your own freeze if versions differ).

- **`requirements.txt`** stays with minimum compatible versions (`>=`) so fresh installs work; the freeze file is the **exact** snapshot for reproducibility.

- **Notebooks:** each training notebook includes an early cell that prints **Python / library versions** and sets **random seeds** (`random`, `numpy`, TensorFlow where applicable). Run notebooks top-to-bottom before archiving.

---

## Ethics, privacy, and limitations

- **Local research software:** data and accounts live in a **local SQLite** file (`air_quality.db`). This is suitable for a supervised thesis demo, not for public deployment without further security review.
- **User content:** uploaded CSVs, forecast history, and optional HelpDesk chat can be **persisted in the database**. In your thesis, state that participants’ data were handled locally, who had access, and that uploads should not contain sensitive personal fields.
- **Known limits (good “future work” bullets):** no formal penetration test; model integrity hashes are optional (`models/hashes.json`); dependency versions should be pinned via `environment-appendix.txt` for archival runs; Streamlit entrypoints rely on [`app/invoke_bootstrap.py`](app/invoke_bootstrap.py) rather than a formal installed package layout.

---

## Database

SQLite file: `air_quality.db` (created automatically on first run).

Tables: `users`, `prediction_history`, `user_settings`, `chat_sessions`, `chat_messages`.

---

## Developer quality checks

```bash
# Lint and format checks
ruff check .
black --check .

# Run tests
pytest -q
```

For thesis documentation or viva prep, a concise **Streamlit smoke checklist** (login, predict, history, settings, HelpDesk) lives in [`MANUAL_TEST_CHECKLIST.md`](MANUAL_TEST_CHECKLIST.md).

CI (GitHub Actions): on push/PR to `main`, [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `ruff`, `black`, `pytest`, and targeted `mypy` (see workflow file for exact commands).
