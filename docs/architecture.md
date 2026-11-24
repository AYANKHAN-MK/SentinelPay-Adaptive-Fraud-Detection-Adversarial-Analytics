# Architecture

```text
Synthetic Events (generate_data.py)
        │
        ▼
DuckDB SQL Feature Engineering (sql/features.sql)
        │
        ▼
Time-Based Train / Validation / Test Split
        │
        ▼
HistGradientBoostingClassifier + Rule Baseline
        │
        ├──► Top-K Analyst Review Metrics
        ├──► Adversarial Evasion Simulator (adversarial.py)
        ├──► PSI Feature Drift Monitor (drift.py)
        └──► Streamlit Workbench (streamlit_app.py) & FastAPI (api/main.py)
```
