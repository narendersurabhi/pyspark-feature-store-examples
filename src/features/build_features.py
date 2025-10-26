#!/usr/bin/env python3
"""
PySpark feature builder (synthetic). Reads claims/providers/members parquet files,
computes simple temporal and ratio features, and writes features.parquet.

Expected inputs:
  data/synth_claims.parquet
  data/providers.parquet
  data/members.parquet
"""
import argparse, pathlib
from pyspark.sql import SparkSession, functions as F, Window

def build(input_dir: str, out_path: str):
    spark = SparkSession.builder.appName("feature-build").getOrCreate()
    base = pathlib.Path(input_dir)

    claims = spark.read.parquet(str(base / "synth_claims.parquet"))
    prov = spark.read.parquet(str(base / "providers.parquet"))
    mem = spark.read.parquet(str(base / "members.parquet"))

    claims = claims.withColumn("dos", F.to_timestamp("dos"))
    claims = claims.withColumn("billed_to_paid", F.col("billed") / (F.col("paid") + F.lit(1e-6)))

    w30 = Window.partitionBy("provider_id").orderBy(F.col("dos").cast("long")).rangeBetween(-30*86400, 0)
    w90 = Window.partitionBy("provider_id").orderBy(F.col("dos").cast("long")).rangeBetween(-90*86400, 0)

    feats = (
        claims
        .withColumn("prov_cnt_30d", F.count("*").over(w30))
        .withColumn("prov_cnt_90d", F.count("*").over(w90))
        .withColumn("avg_ratio_30d", F.avg("billed_to_paid").over(w30))
        .withColumn("avg_ratio_90d", F.avg("billed_to_paid").over(w90))
        .withColumn("after_hours", (F.hour("dos").between(0,7) | F.hour("dos").between(19,23)).cast("int"))
        .join(prov, "provider_id", "left")
        .join(mem.select("member_id","age"), "member_id", "left")
        .select(
            "claim_id","provider_id","member_id","proc_code","units","dos","billed","paid",
            "billed_to_paid","prov_cnt_30d","prov_cnt_90d","avg_ratio_30d","avg_ratio_90d",
            "after_hours","specialty","age"
        )
    )

    feats.write.mode("overwrite").parquet(out_path)
    print(f"Wrote {out_path}")
    spark.stop()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data", help="folder with parquet inputs")
    ap.add_argument("--out", default="features.parquet")
    args = ap.parse_args()
    build(args.input, args.out)

if __name__ == "__main__":
    main()
