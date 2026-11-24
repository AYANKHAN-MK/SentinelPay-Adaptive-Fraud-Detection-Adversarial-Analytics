import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelpay.generate_data import generate
from sentinelpay.build_features import build_feature_df
from sentinelpay.train import train_pipeline


def test_end_to_end_pipeline():
    raw = generate(2000, seed=7)
    assert len(raw) == 2000
    feats = build_feature_df(raw)
    assert "account_txn_1h" in feats.columns
    res = train_pipeline(feats, review_fraction=0.02, seed=7)
    assert 0.0 <= res["metrics"]["pr_auc"] <= 1.0
