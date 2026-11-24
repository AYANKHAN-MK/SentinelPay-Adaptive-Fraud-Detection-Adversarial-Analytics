from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

FEATURES = [
    "amount",
    "failed_attempts",
    "is_new_account",
    "is_shared_device",
    "is_shared_ip",
    "account_txn_1h",
    "account_txn_24h",
    "account_amount_24h",
    "device_txn_1h",
    "ip_txn_1h",
    "accounts_per_device",
    "accounts_per_ip",
]


def top_k_metrics(y: np.ndarray, score: np.ndarray, fraction: float = 0.01) -> dict:
    k = max(1, int(len(y) * fraction))
    order = np.argsort(-score)[:k]
    yy = np.asarray(y)[order]
    return {
        "review_rate": fraction,
        "precision_at_top_k": float(yy.mean()),
        "recall_at_top_k": float(yy.sum() / max(np.asarray(y).sum(), 1)),
    }


def rule_baseline_score(df: pd.DataFrame) -> np.ndarray:
    amt_p95 = df["amount"].quantile(0.95)
    raw = (
        (df["amount"] > amt_p95).astype(float) * 0.25
        + (df["failed_attempts"] >= 2).astype(float) * 0.20
        + (df["account_txn_1h"] >= 3).astype(float) * 0.20
        + (df["accounts_per_device"] >= 3).astype(float) * 0.20
        + (df["is_shared_device"] == 1).astype(float) * 0.15
    )
    return np.clip(raw.to_numpy(), 0.0, 1.0)


def train_pipeline(
    features_df: pd.DataFrame,
    review_fraction: float = 0.01,
    seed: int = 42,
) -> dict:
    df = features_df.sort_values("event_time").reset_index(drop=True)
    cut = int(len(df) * 0.70)
    val_cut = int(len(df) * 0.85)
    train_df = df.iloc[:cut]
    test_df = df.iloc[val_cut:].copy()

    X_train, y_train = train_df[FEATURES], train_df["is_fraud"].to_numpy()
    X_test, y_test = test_df[FEATURES], test_df["is_fraud"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.08,
        max_leaf_nodes=24,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    ml_scores = model.predict_proba(X_test)[:, 1]
    rule_scores = rule_baseline_score(test_df)

    threshold = float(np.quantile(ml_scores, 1.0 - review_fraction))
    ml_pred = (ml_scores >= threshold).astype(int)

    rule_threshold = float(np.quantile(rule_scores, 1.0 - review_fraction))
    rule_pred = (rule_scores >= rule_threshold).astype(int)

    sample_n = min(3000, len(X_test))
    perm = permutation_importance(
        model,
        X_test.iloc[:sample_n],
        y_test[:sample_n],
        n_repeats=3,
        random_state=seed,
        scoring="average_precision",
    )
    importance_df = pd.DataFrame(
        {"feature": FEATURES, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)

    test_df["ml_score"] = ml_scores
    test_df["rule_score"] = rule_scores
    test_df["ml_pred"] = ml_pred

    metrics = {
        "pr_auc": float(average_precision_score(y_test, ml_scores)),
        "rule_pr_auc": float(average_precision_score(y_test, rule_scores)),
        "precision": float(precision_score(y_test, ml_pred, zero_division=0)),
        "recall": float(recall_score(y_test, ml_pred, zero_division=0)),
        "rule_precision": float(precision_score(y_test, rule_pred, zero_division=0)),
        "rule_recall": float(recall_score(y_test, rule_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, ml_pred).tolist(),
        **top_k_metrics(y_test, ml_scores, fraction=review_fraction),
        "threshold": threshold,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_fraud_rate": float(y_test.mean()),
    }

    return {
        "model": model,
        "features": FEATURES,
        "threshold": threshold,
        "metrics": metrics,
        "test_df": test_df,
        "importance_df": importance_df,
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    res = train_pipeline(df)
    joblib.dump(
        {"model": res["model"], "features": res["features"], "threshold": res["threshold"]},
        MODEL_DIR / "fraud_model.joblib",
    )
    (MODEL_DIR / "metrics.json").write_text(json.dumps(res["metrics"], indent=2), encoding="utf-8")
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()
