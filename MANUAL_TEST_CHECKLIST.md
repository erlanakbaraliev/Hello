# Manual test checklist (thesis appendix)

Short scenarios to run after `streamlit run app/Home.py` with a fresh or test account. Tick when verified.

## Authentication

- [ ] Register a new user (matching username rules); confirm redirect / signed-in state.
- [ ] Log out; log in with correct credentials.
- [ ] Log in with wrong password; expect “Invalid username or password.”
- [ ] After several wrong passwords (lockout threshold), expect a **locked account** message with a retry time, not only the generic invalid-credentials text.

## Predict future quality

- [ ] Upload a valid hourly CSV (training schema); run **ARIMA**, **LSTM**, and **XGBoost** in turn; confirm chart shows history plus **168-hour** forecast.
- [ ] Confirm chart title / page copy refers to hourly horizon (not ambiguous “daily seven-day” wording).

## Settings ↔ Predict

- [ ] In **Settings**, set default model to e.g. XGBoost; save.
- [ ] Open **Predict future quality**; confirm the model selectbox defaults to that choice.

## History

- [ ] After a successful prediction, open **History**; filter by date/model; download CSV; delete one row if appropriate.

## HelpDesk

- [ ] Without API key: sensible message (no crash).
- [ ] With key configured: send a short question; confirm a reply appears (or graceful timeout/error).

## Data Explorer & Dashboard (smoke)

- [ ] **Data Explorer**: upload a small numeric CSV; charts render.
- [ ] **Dashboard**: loads default London view without error when signed in.
