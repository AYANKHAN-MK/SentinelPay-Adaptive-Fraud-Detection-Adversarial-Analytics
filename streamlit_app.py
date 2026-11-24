import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import streamlit as st

from sentinelpay.generate_data import generate
from sentinelpay.build_features import build_feature_df
from sentinelpay.train import train_pipeline, FEATURES
from sentinelpay.adversarial import run_adversarial_scenarios
from sentinelpay.drift import compute_drift_report

st.set_page_config(
    page_title="SentinelPay | Adaptive Fraud Detection & Adversarial Analytics",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(show_spinner="Generating synthetic transactions & engineering DuckDB SQL features...")
def load_pipeline(n_rows: int, seed: int, review_pct: float):
    raw_df = generate(rows=n_rows, seed=seed)
    features_df = build_feature_df(raw_df)
    results = train_pipeline(features_df, review_fraction=review_pct / 100.0, seed=seed)
    return raw_df, features_df, results


st.title("🛡️ SentinelPay — Adaptive Fraud Detection & Adversarial Analytics")
st.caption(
    "End-to-end fraud analytics workbench combining DuckDB SQL behavioural features, "
    "gradient-boosted risk scoring, adversarial evasion simulations, and PSI drift monitoring."
)

# Sidebar Controls
with st.sidebar:
    st.header("Pipeline Controls")
    n_rows = st.slider("Synthetic Transactions", min_value=10000, max_value=100000, value=30000, step=5000)
    seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
    review_pct = st.slider("Analyst Review Capacity (Top %)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
    st.divider()
    st.markdown(
    "**Pipeline Stack**\n"
    "- **Data:** Synthetic payment & device graph\n"
    "- **Features:** DuckDB window & entity SQL\n"
    "- **Model:** HistGradientBoostingClassifier\n"
    "- **Monitoring:** Population Stability Index (PSI)"
)
raw_df, features_df, results = load_pipeline(n_rows, int(seed), float(review_pct))
metrics = results["metrics"]
model = results["model"]
threshold = results["threshold"]
test_df = results["test_df"]
importance_df = results["importance_df"]

# Top KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{len(features_df):,}")
c2.metric("Test Fraud Rate", f"{metrics['test_fraud_rate']:.2%}")
c3.metric(
    "ML PR-AUC",
    f"{metrics['pr_auc']:.3f}",
    delta=f"{metrics['pr_auc'] - metrics['rule_pr_auc']:+.3f} vs Rules",
)
c4.metric(
    f"Precision @ Top {review_pct:.1f}%",
    f"{metrics['precision_at_top_k']:.1%}",
    delta=f"{metrics['precision'] - metrics['rule_precision']:+.1%} vs Rules",
)
c5.metric(
    f"Recall @ Top {review_pct:.1f}%",
    f"{metrics['recall_at_top_k']:.1%}",
    delta=f"{metrics['recall'] - metrics['rule_recall']:+.1%} vs Rules",
)

tab_overview, tab_adv, tab_drift, tab_scorer, tab_sql = st.tabs(
    [
        "📊 Detection Performance",
        "⚔️ Adversarial Testing",
        "📉 Feature Drift (PSI)",
        "🔍 Live Transaction Scorer",
        "🧾 SQL & Investigation",
    ]
)

# TAB 1: Detection Performance
with tab_overview:
    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        st.subheader("Permutation Feature Importance (PR-AUC Drop)")
        chart_imp = importance_df.set_index("feature")["importance"]
        st.bar_chart(chart_imp)

    with col_right:
        st.subheader("Rule Baseline vs. ML Model Comparison")
        comp_df = pd.DataFrame(
            {
                "Detector": ["Rule-Based Baseline", "Gradient-Boosted ML"],
                "PR-AUC": [round(metrics["rule_pr_auc"], 4), round(metrics["pr_auc"], 4)],
                f"Precision @ Top {review_pct:.1f}%": [
                    round(metrics["rule_precision"], 4),
                    round(metrics["precision"], 4),
                ],
                f"Recall @ Top {review_pct:.1f}%": [
                    round(metrics["rule_recall"], 4),
                    round(metrics["recall"], 4),
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legitimate (0)", "Actual Fraud (1)"],
            columns=["Predicted Allow (0)", "Predicted Review (1)"],
        )
        st.subheader("Confusion Matrix at Operating Threshold")
        st.dataframe(cm_df, use_container_width=True)

    st.subheader("Highest-Risk Flagged Transactions (Test Window)")
    top_flagged = (
        test_df.sort_values("ml_score", ascending=False)
        .head(15)[
            [
                "transaction_id",
                "event_time",
                "account_id",
                "device_id",
                "amount",
                "merchant_type",
                "country",
                "account_txn_1h",
                "accounts_per_device",
                "failed_attempts",
                "ml_score",
                "is_fraud",
            ]
        ]
        .copy()
    )
    top_flagged["ml_score"] = top_flagged["ml_score"].round(4)
    st.dataframe(top_flagged, use_container_width=True, hide_index=True)

# TAB 2: Adversarial Testing
with tab_adv:
    st.subheader("Simulate Adaptive Fraud Evasion")
    st.write(
        "Test how much detection recall drops when fraudsters learn system thresholds and adapt their behaviour."
    )
    ac1, ac2, ac3, ac4 = st.columns(4)
    amount_scale = ac1.slider("Amount Multiplier (Splitting)", 0.30, 1.00, 0.70, 0.05)
    velocity_scale = ac2.slider("Velocity Multiplier (Throttling)", 0.20, 1.00, 0.55, 0.05)
    device_rotation = ac3.checkbox("Rotate Devices & IPs", value=True)
    cap_failed = ac4.checkbox("Cap Failed Attempts <= 1", value=True)

    adv_summary, fraud_eval = run_adversarial_scenarios(
        model=model,
        features=FEATURES,
        threshold=threshold,
        test_df=test_df,
        amount_scale=amount_scale,
        velocity_scale=velocity_scale,
        device_rotation=device_rotation,
        cap_failed_attempts=cap_failed,
        seed=int(seed),
    )

    if not adv_summary.empty:
        st.dataframe(adv_summary, use_container_width=True, hide_index=True)
        st.bar_chart(adv_summary.set_index("Scenario")["Fraud Recall"])

        st.subheader("Hardest-to-Detect Adapted Fraud Cases (Largest Score Drops)")
        evaded = (
            fraud_eval.sort_values("score_drop", ascending=False)
            .head(10)[
                [
                    "transaction_id",
                    "amount",
                    "account_txn_1h",
                    "accounts_per_device",
                    "failed_attempts",
                    "baseline_score",
                    "evasion_score",
                    "score_drop",
                ]
            ]
            .round(4)
        )
        st.dataframe(evaded, use_container_width=True, hide_index=True)

# TAB 3: Feature Drift (PSI)
with tab_drift:
    st.subheader("Population Stability Index (PSI) Monitoring")
    st.write(
        "Compares the early reference window (first 75% of timeline) against the latest window (last 25%). "
        "Inject synthetic distribution shifts below to test drift alerting."
    )
    dc1, dc2 = st.columns(2)
    amount_shift = dc1.slider("Current Window Amount Shift Multiplier", 0.50, 2.50, 1.00, 0.05)
    velocity_shift = dc2.slider("Current Window Velocity Shift Multiplier", 0.50, 2.50, 1.00, 0.05)

    drift_df = compute_drift_report(features_df, amount_shift=amount_shift, velocity_shift=velocity_shift)
    st.dataframe(drift_df, use_container_width=True, hide_index=True)
    st.bar_chart(drift_df.set_index("Feature")["PSI"])

# TAB 4: Live Transaction Scorer
with tab_scorer:
    st.subheader("Interactive Risk Scoring Simulator")
    st.write("Adjust transaction and behavioural signals below to inspect real-time model scoring.")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_amount = st.number_input("Transaction Amount", min_value=1.0, max_value=100000.0, value=1450.0, step=50.0)
        s_failed = st.number_input("Failed Attempts", min_value=0, max_value=20, value=2, step=1)
        s_new_acc = st.selectbox("Is New Account?", [0, 1], index=1)
        s_shared_dev = st.selectbox("Is Shared Device?", [0, 1], index=1)
    with sc2:
        s_shared_ip = st.selectbox("Is Shared IP?", [0, 1], index=0)
        s_acc_1h = st.number_input("Account Txns (1h)", min_value=1, max_value=100, value=4, step=1)
        s_acc_24h = st.number_input("Account Txns (24h)", min_value=1, max_value=500, value=9, step=1)
        s_acc_amt_24h = st.number_input("Account Amount (24h)", min_value=1.0, max_value=500000.0, value=6200.0, step=100.0)
    with sc3:
        s_dev_1h = st.number_input("Device Txns (1h)", min_value=1, max_value=100, value=4, step=1)
        s_ip_1h = st.number_input("IP Txns (1h)", min_value=1, max_value=100, value=3, step=1)
        s_acc_per_dev = st.number_input("Accounts per Device", min_value=1, max_value=50, value=3, step=1)
        s_acc_per_ip = st.number_input("Accounts per IP", min_value=1, max_value=50, value=2, step=1)

    sample_row = pd.DataFrame(
        [
            {
                "amount": float(s_amount),
                "failed_attempts": int(s_failed),
                "is_new_account": int(s_new_acc),
                "is_shared_device": int(s_shared_dev),
                "is_shared_ip": int(s_shared_ip),
                "account_txn_1h": int(s_acc_1h),
                "account_txn_24h": int(s_acc_24h),
                "account_amount_24h": float(s_acc_amt_24h),
                "device_txn_1h": int(s_dev_1h),
                "ip_txn_1h": int(s_ip_1h),
                "accounts_per_device": int(s_acc_per_dev),
                "accounts_per_ip": int(s_acc_per_ip),
            }
        ]
    )[FEATURES]

    prob = float(model.predict_proba(sample_row)[0, 1])
    decision = "🚨 REVIEW (Flagged)" if prob >= threshold else "✅ ALLOW"

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Predicted Fraud Probability", f"{prob:.2%}")
    rc2.metric("Operating Threshold", f"{threshold:.2%}")
    rc3.metric("Decision", decision)

# TAB 5: SQL & Investigation Workflow
with tab_sql:
    st.subheader("DuckDB SQL Feature Engineering Layer (`sql/features.sql`)")
    st.code((ROOT / "sql" / "features.sql").read_text(encoding="utf-8"), language="sql")
    st.subheader("Analyst Investigation Playbook")
    st.markdown((ROOT / "docs" / "fraud_investigation.md").read_text(encoding="utf-8"))
