# PySpark Feature Store Examples — Healthcare Style (Synthetic)

Contents
- Slowly changing provider attributes (SCD2).
- Temporal windows (30/90/180d) with watermarking.
- Network features: provider–member co-occurrence signals.
- Data quality checks and expectations.

Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/synthetic_claims.py
python src/features/build_features.py --input data --out features.parquet
python src/quality/run_checks.py --input features.parquet

Outputs
- `features.parquet`
- `reports/quality.html`

Disclaimer
Synthetic data only. No employer IP.
