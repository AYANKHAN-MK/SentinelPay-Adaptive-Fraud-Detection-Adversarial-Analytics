# Fraud Investigation Workflow

When a transaction crosses the top-K review threshold, follow a structured triage check:

1. **Velocity Check:** Compare `account_txn_1h` and `account_txn_24h` against the account's historical baseline.
2. **Amount Anomaly:** Check whether `amount` or `account_amount_24h` spikes sharply or splits into repeated smaller amounts.
3. **Entity Graph:** Inspect `accounts_per_device` and `accounts_per_ip` to spot shared infrastructure clusters.
4. **Retry Bursts:** Check `failed_attempts` immediately preceding a successful charge (credential or card testing).
5. **Feedback Loop:** Record confirmed fraud/legitimate outcomes to retrain thresholds and monitor PSI drift.
