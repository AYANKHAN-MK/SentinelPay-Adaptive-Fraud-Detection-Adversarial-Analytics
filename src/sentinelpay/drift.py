from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MONITORED_FEATURES = [
    "amount",
    "account_txn_1h",
    "account_txn_24h",
    "device_txn_1h",
    "ip_txn_1h",
    "accounts_per_device",
    "accounts_per_ip",
    "failed_attempts",
]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_drift_report(
    features_df: pd.DataFrame,
    amount_shift: float = 1.0,
    velocity_shift: float = 1.0,
) -> pd.DataFrame:
    df = features_df.sort_values("event_time").reset_index(drop=True)
    split = int(len(df) * 0.75)
    ref = df.iloc[:split]
    current = df.iloc[split:].copy()

    current["amount"] = current["amount"] * amount_shift
    current["account_txn_1h"] = np.maximum(1, np.round(current["account_txn_1h"] * velocity_shift))
    current["device_txn_1h"] = np.maximum(1, np.round(current["device_txn_1h"] * velocity_shift))

    rows = []
    for col in MONITORED_FEATURES:
        val = psi(ref[col].to_numpy(), current[col].to_numpy())
        status = "ALERT (>=0.20)" if val >= 0.20 else ("WATCH (0.10-0.19)" if val >= 0.10 else "OK (<0.10)")
        rows.append(
            {
                "Feature": col,
                "PSI": round(val, 4),
                "Status": status,
                "Ref Mean": round(float(ref[col].mean()), 3),
                "Current Mean": round(float(current[col].mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def main():
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    print(compute_drift_report(df).to_string(index=False))


if __name__ == "__main__":
    main()
