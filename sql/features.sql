WITH base AS (
    SELECT *,
           date_trunc('hour', event_time) AS event_hour,
           date_trunc('day', event_time) AS event_day
    FROM raw_txns
),
velocity AS (
    SELECT
        a.transaction_id,
        COUNT(*) OVER (
            PARTITION BY a.account_id, a.event_hour
        ) AS account_txn_1h,
        COUNT(*) OVER (
            PARTITION BY a.account_id
            ORDER BY a.event_time
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW
        ) AS account_txn_24h,
        SUM(a.amount) OVER (
            PARTITION BY a.account_id
            ORDER BY a.event_time
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW
        ) AS account_amount_24h,
        COUNT(*) OVER (
            PARTITION BY a.device_id, a.event_hour
        ) AS device_txn_1h,
        COUNT(*) OVER (
            PARTITION BY a.ip_id, a.event_hour
        ) AS ip_txn_1h
    FROM base a
),
relationships AS (
    SELECT
        transaction_id,
        COUNT(DISTINCT account_id) OVER (PARTITION BY device_id) AS accounts_per_device,
        COUNT(DISTINCT account_id) OVER (PARTITION BY ip_id) AS accounts_per_ip
    FROM base
)
SELECT
    b.transaction_id,
    b.event_time,
    b.account_id,
    b.device_id,
    b.ip_id,
    b.amount,
    b.failed_attempts,
    b.is_new_account,
    b.is_shared_device,
    b.is_shared_ip,
    b.merchant_type,
    b.country,
    b.is_fraud,
    v.account_txn_1h,
    v.account_txn_24h,
    v.account_amount_24h,
    v.device_txn_1h,
    v.ip_txn_1h,
    r.accounts_per_device,
    r.accounts_per_ip
FROM base b
JOIN velocity v USING (transaction_id)
JOIN relationships r USING (transaction_id)
