# SentinelPay — Adaptive Fraud Detection & Adversarial Analytics

SentinelPay is an end-to-end fraud analytics workbench built to examine how payment fraud detectors behave when attackers adapt to static rules and thresholds.

It combines synthetic transaction generation, DuckDB SQL behavioural feature engineering, gradient-boosted risk scoring, adversarial evasion testing, PSI drift monitoring, and an interactive Streamlit dashboard.

## Live App & Entry Point

- **Streamlit Main File:** `streamlit_app.py`
- **Optional REST API:** `api/main.py` (`uvicorn api.main:app --reload`)

## Project Structure

```text
sentinelpay/
├── streamlit_app.py              # Interactive Streamlit dashboard (main entry point)
├── requirements.txt              # Python dependencies compatible with Streamlit Cloud
├── .streamlit/
│   └── config.toml               # Streamlit UI theme & server settings
├── src/
│   └── sentinelpay/
│       ├── generate_data.py      # Synthetic payment, device, IP & fraud cluster generator
│       ├── build_features.py     # DuckDB SQL feature engineering runner
│       ├── train.py              # Time-split HistGradientBoosting model & rule baseline
│       ├── adversarial.py        # Adaptive evasion scenarios (splitting, throttling, rotation)
│       └── drift.py              # Population Stability Index (PSI) drift checks
├── sql/
│   └── features.sql              # Windowed velocity & shared-entity SQL queries
├── api/
│   └── main.py                   # FastAPI scoring endpoint
├── docs/
│   ├── architecture.md           # System pipeline diagram
│   └── fraud_investigation.md    # Analyst triage workflow
└── tests/
    └── test_smoke.py             # End-to-end pipeline smoke test
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push the contents of this folder to your GitHub repository.
2. In Streamlit Community Cloud, click **New app**.
3. Select your repository and branch (`main`), and set **Main file path** to `streamlit_app.py`.
4. Click **Deploy**.
