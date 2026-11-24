from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
BUNDLE = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
app = FastAPI(title="SentinelPay Fraud Scoring API", version="0.2.0")


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    failed_attempts: int = Field(ge=0)
    is_new_account: int = Field(ge=0, le=1)
    is_shared_device: int = Field(ge=0, le=1)
    is_shared_ip: int = Field(ge=0, le=1)
    account_txn_1h: int = Field(ge=0)
    account_txn_24h: int = Field(ge=0)
    account_amount_24h: float = Field(ge=0)
    device_txn_1h: int = Field(ge=0)
    ip_txn_1h: int = Field(ge=0)
    accounts_per_device: int = Field(ge=1)
    accounts_per_ip: int = Field(ge=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None}


@app.post("/score")
def score(txn: Transaction):
    if BUNDLE is None:
        return {"error": "Model not trained. Run: PYTHONPATH=src python -m sentinelpay.train"}
    x = pd.DataFrame([txn.model_dump()])[BUNDLE["features"]]
    p = float(BUNDLE["model"].predict_proba(x)[0, 1])
    return {
        "fraud_probability": round(p, 6),
        "decision": "review" if p >= BUNDLE["threshold"] else "allow",
    }
