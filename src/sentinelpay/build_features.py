from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build_feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    sql = (ROOT / "sql" / "features.sql").read_text(encoding="utf-8")
    con = duckdb.connect()
    con.register("raw_txns", raw_df)
    features_df = con.execute(sql).df()
    con.close()
    return features_df.sort_values("event_time").reset_index(drop=True)


def main():
    raw_df = pd.read_parquet(ROOT / "data" / "transactions.parquet")
    features_df = build_feature_df(raw_df)
    out = ROOT / "data" / "features.parquet"
    features_df.to_parquet(out, index=False)
    print(f"wrote {len(features_df):,} feature rows to {out}")


if __name__ == "__main__":
    main()
