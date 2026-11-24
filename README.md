# SentinelPay — Adaptive Fraud Detection & Adversarial Analytics

SentinelPay is an end-to-end fraud analytics workbench built to examine how payment fraud detectors behave when attackers adapt to static rules and thresholds.

It combines synthetic transaction generation, DuckDB SQL behavioural feature engineering, gradient-boosted risk scoring, adversarial evasion testing, PSI drift monitoring, and an interactive Streamlit dashboard.

## Live App & Entry Point

- **Live Streamlit App:** https://sentinelpay-adaptive-fraud-detection-adversarial-analytics-5xa.streamlit.app/
- **Streamlit Main File:** `streamlit_app.py`
- **Optional REST API:** `api/main.py` (`uvicorn api.main:app --reload`)

## Project Structure

```text
sentinelpay/
├── streamlit_app.py              # Interactive Streamlit dashboard (main entry point)[cite: 16]
├── requirements.txt              # Python dependencies compatible with Streamlit Cloud[cite: 16]
├── .streamlit/
│   └── config.toml               # Streamlit UI theme & server settings[cite: 16]
├── src/
│   └── sentinelpay/
│       ├── generate_data.py      # Synthetic payment, device, IP & fraud cluster generator[cite: 16]
│       ├── build_features.py     # DuckDB SQL feature engineering runner[cite: 16]
│       ├── train.py              # Time-split HistGradientBoosting model & rule baseline[cite: 16]
│       ├── adversarial.py        # Adaptive evasion scenarios (splitting, throttling, rotation)[cite: 16]
│       └── drift.py              # Population Stability Index (PSI) drift checks[cite: 16]
├── sql/
│   └── features.sql              # Windowed velocity & shared-entity SQL queries[cite: 16]
├── api/
│   └── main.py                   # FastAPI scoring endpoint[cite: 16]
├── docs/
│   ├── architecture.md           # System pipeline diagram[cite: 16]
│   └── fraud_investigation.md    # Analyst triage workflow[cite: 16]
└── tests/
    └── test_smoke.py             # End-to-end pipeline smoke test[cite: 16]
