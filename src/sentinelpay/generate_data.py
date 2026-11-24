import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def generate(rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_accounts = max(2500, rows // 30)
    n_devices = max(1800, rows // 45)
    n_ips = max(2500, rows // 35)

    account = rng.integers(1, n_accounts + 1, rows)
    device = rng.integers(1, n_devices + 1, rows)
    ip = rng.integers(1, n_ips + 1, rows)
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 90 * 86400, rows), unit="s")
    amount = np.round(np.exp(rng.normal(6.0, 0.75, rows)), 2)
    merchant = rng.choice(
        ["billpay", "rewards", "wallet", "travel", "shopping", "subscription"],
        rows,
        p=[0.25, 0.12, 0.12, 0.10, 0.25, 0.16],
    )
    country = rng.choice(["IN", "AE", "SG", "US"], rows, p=[0.91, 0.03, 0.03, 0.03])
    failed = rng.poisson(0.25, rows)

    fraud_accounts = set(
        rng.choice(np.arange(1, n_accounts + 1), size=max(50, n_accounts // 50), replace=False)
    )
    fraud = np.isin(account, list(fraud_accounts))

    new_account = rng.random(rows) < 0.07
    shared_device = rng.random(rows) < 0.08
    shared_ip = rng.random(rows) < 0.10
    high_fail = failed >= 2
    unusual_amount = amount > np.quantile(amount, 0.97)

    latent = (
        fraud.astype(float) * 2.3
        + new_account.astype(float) * 0.45
        + shared_device.astype(float) * 0.55
        + shared_ip.astype(float) * 0.35
        + high_fail.astype(float) * 0.40
        + unusual_amount.astype(float) * 0.35
        + (country != "IN").astype(float) * 0.20
        + rng.normal(0, 0.65, rows)
    )
    probability = 1 / (1 + np.exp(-(latent - 2.6)))
    is_fraud = rng.random(rows) < probability

    amount = np.where(is_fraud, amount * rng.choice([1.0, 0.55, 1.8], rows, p=[0.55, 0.25, 0.20]), amount)
    amount = np.round(np.maximum(amount, 1.0), 2)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, rows + 1),
            "event_time": ts,
            "account_id": account,
            "device_id": device,
            "ip_id": ip,
            "amount": amount,
            "merchant_type": merchant,
            "country": country,
            "failed_attempts": failed,
            "is_new_account": new_account.astype(int),
            "is_shared_device": shared_device.astype(int),
            "is_shared_ip": shared_ip.astype(int),
            "is_fraud": is_fraud.astype(int),
        }
    ).sort_values("event_time")
    return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    out = DATA / "transactions.parquet"
    df = generate(args.rows, args.seed)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows to {out} (fraud rate: {df.is_fraud.mean():.3%})")


if __name__ == "__main__":
    main()
