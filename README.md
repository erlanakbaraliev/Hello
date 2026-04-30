# Urban Air Quality Prediction Platform

A data-driven machine learning platform for predicting urban air quality (PM2.5) and assessing pollution risk, built with Streamlit, LSTM, XGBoost, and ARIMA.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run front_end/Home.py
```

## Project Structure

```
├── front_end/          UI layer (Streamlit pages, theme, auth)
│   ├── Home.py         Entry point
│   ├── ui.py           Shared layout & session management
│   ├── pages/          Streamlit multi-page scripts
│   └── static/         CSS theme
├── back_end/           Business logic
│   ├── forecasting.py  Model loading & 168-hour forecast dispatch
│   ├── data_processing.py  Data validation & preprocessing
│   ├── page_dashboard.py   Dashboard chart logic
│   ├── page_predict.py     Forecast figure builder
│   ├── page_helpdesk.py    Chat UI logic
│   └── helpdesk_gemini.py  Gemini API client
├── db/                 Database layer
│   └── database.py     SQLite: users, settings, history, chat, auth tokens
├── models/             ML models
│   ├── models.py       Model metadata helpers
│   ├── artifacts/      Saved .pkl, .keras, .joblib files
│   └── notebooks/      Training notebooks
├── data/               Datasets & preprocessing scripts
│   └── london_2024.csv Primary dataset (hourly, full year)
├── tests/              Pytest test suite
├── thesis-doc/         LaTeX thesis documentation (ELTE template)
├── requirements.txt
└── pyproject.toml
```

## Pages

| Page | Purpose |
|------|---------|
| Home | Authentication (login/register) + project overview |
| Data Explorer | Upload CSV, view correlations & distributions |
| Predict | 168-hour PM2.5 forecast (LSTM / XGBoost) |
| Dashboard | Live metrics, AQI, pollutant trends |
| History | Browse & download past predictions |
| Settings | Profile, preferences, security |
| HelpDesk | Gemini-powered Q&A |

## Models

Three forecasting approaches on hourly multivariate London air quality data:

- **ARIMA** — univariate baseline, captures temporal autocorrelation
- **LSTM** — recurrent neural network, multivariate, captures long-range dependencies
- **XGBoost** — gradient-boosted trees with engineered lag features

All produce **168-hour** (one-week) recursive forecasts. Train via notebooks:

```bash
jupyter notebook models/notebooks/
```

Artifacts are saved to `models/artifacts/`.

## Configuration

Optional Gemini API key for HelpDesk:

```bash
mkdir -p front_end/.streamlit
echo 'GEMINI_API_KEY = "your-key"' > front_end/.streamlit/secrets.toml
```

## Development

```bash
ruff check .
black --check .
pytest -q
```

## Tech Stack

Python, Streamlit, pandas, Plotly, TensorFlow/Keras, XGBoost, statsmodels, scikit-learn, SQLite, bcrypt

## Author

Musaev Kutmanbek — ELTE Faculty of Informatics, Computer Science BSc  
Supervisor: Szabó László Ferenc (Associate Professor and Chair, Dept. of Algorithms and Applications)
