from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parents[2]


def run_adversarial_scenarios(
    model,
    features: list[str],
    threshold: float,
    test_df: pd.DataFrame,
    amount_scale: float = 0.75,
    velocity_scale: float = 0.55,
    device_rotation: bool = True,
    cap_failed_attempts: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fraud = test_df[test_df["is_fraud"] == 1].copy()
    if fraud.empty:
        return pd.DataFrame(), pd.DataFrame()

    y_true = fraud["is_fraud"].to_numpy()
    base_scores = model.predict_proba(fraud[features])[:, 1]
    base_recall = float(recall_score(y_true, (base_scores >= threshold).astype(int)))

    scenarios = {}

    # 1. Baseline
    scenarios["Baseline (No Adaptation)"] = fraud.copy()

    # 2. Amount splitting / normalization
    s_amt = fraud.copy()
    s_amt["amount"] = np.maximum(1.0, s_amt["amount"] * amount_scale)
    s_amt["account_amount_24h"] = np.maximum(1.0, s_amt["account_amount_24h"] * amount_scale)
    scenarios["Amount Splitting"] = s_amt

    # 3. Lower velocity & retry throttling
    s_vel = fraud.copy()
    s_vel["account_txn_1h"] = np.maximum(1, np.round(s_vel["account_txn_1h"] * velocity_scale))
    s_vel["account_txn_24h"] = np.maximum(1, np.round(s_vel["account_txn_24h"] * velocity_scale))
    s_vel["device_txn_1h"] = np.maximum(1, np.round(s_vel["device_txn_1h"] * velocity_scale))
    s_vel["ip_txn_1h"] = np.maximum(1, np.round(s_vel["ip_txn_1h"] * velocity_scale))
    if cap_failed_attempts:
        s_vel["failed_attempts"] = np.minimum(s_vel["failed_attempts"], 1)
    scenarios["Lower Velocity & Throttled Retries"] = s_vel

    # 4. Device / IP rotation
    s_dev = fraud.copy()
    if device_rotation:
        s_dev["is_shared_device"] = 0
        s_dev["is_shared_ip"] = 0
        s_dev["accounts_per_device"] = 1
        s_dev["accounts_per_ip"] = 1
    scenarios["Device & IP Rotation"] = s_dev

    # 5. Combined adaptation
    s_comb = fraud.copy()
    s_comb["amount"] = np.maximum(1.0, s_comb["amount"] * rng.uniform(max(0.4, amount_scale - 0.1), min(1.05, amount_scale + 0.1), len(s_comb)))
    s_comb["account_amount_24h"] = np.maximum(1.0, s_comb["account_amount_24h"] * amount_scale)
    s_comb["account_txn_1h"] = np.maximum(1, np.round(s_comb["account_txn_1h"] * velocity_scale))
    s_comb["account_txn_24h"] = np.maximum(1, np.round(s_comb["account_txn_24h"] * velocity_scale))
    s_comb["device_txn_1h"] = np.maximum(1, np.round(s_comb["device_txn_1h"] * velocity_scale))
    s_comb["ip_txn_1h"] = np.maximum(1, np.round(s_comb["ip_txn_1h"] * velocity_scale))
    if cap_failed_attempts:
        s_comb["failed_attempts"] = np.minimum(s_comb["failed_attempts"], 1)
    if device_rotation:
        s_comb["is_shared_device"] = 0
        s_comb["is_shared_ip"] = 0
        s_comb["accounts_per_device"] = 1
        s_comb["accounts_per_ip"] = 1
    scenarios["Combined Evasion"] = s_comb

    rows = []
    for name, df_scen in scenarios.items():
        sc = model.predict_proba(df_scen[features])[:, 1]
        rec = float(recall_score(y_true, (sc >= threshold).astype(int)))
        rows.append(
            {
                "Scenario": name,
                "Fraud Recall": round(rec, 4),
                "Recall Drop vs Baseline": round(rec - base_recall, 4),
                "Mean Risk Score": round(float(sc.mean()), 4),
                "Median Risk Score": round(float(np.median(sc)), 4),
            }
        )

    combined_scores = model.predict_proba(s_comb[features])[:, 1]
    fraud_eval = fraud.copy()
    fraud_eval["baseline_score"] = base_scores
    fraud_eval["evasion_score"] = combined_scores
    fraud_eval["score_drop"] = fraud_eval["baseline_score"] - fraud_eval["evasion_score"]

    return pd.DataFrame(rows), fraud_eval


def main():
    bundle = joblib.load(ROOT / "models" / "fraud_model.joblib")
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    summary_df, _ = run_adversarial_scenarios(bundle["model"], bundle["features"], bundle["threshold"], df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
