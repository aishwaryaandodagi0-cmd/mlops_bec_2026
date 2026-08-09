"""
monitoring/generate_report.py — Evidently AI Drift Report (Session 9)
=======================================================================
USAGE:
    python monitoring/generate_report.py

WHAT THIS DOES:
    1. Loads the reference dataset (original training data)
    2. Simulates a "current" production dataset (sampled + optionally drifted)
    3. Generates an Evidently HTML drift report comparing the two
    4. Checks the drift share against a threshold — exits 1 if exceeded

EXIT CODES:
    0 → drift within acceptable range
    1 → drift threshold exceeded (used by monitoring/check_and_retrain.py
        and can be wired into a scheduled CI job)

OUTPUT:
    monitoring/reports/drift_report.html — open this in any browser
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report


FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def build_drift_batch(n: int, seed: int = 99) -> pd.DataFrame:
    """
    Simulate a batch of 'new customer segment' data representing drift:
    senior citizens, short tenure, high monthly charges, month-to-month
    contracts — a demographic and behavioural shift from the training set.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gender":           rng.integers(0, 2, n),
        "SeniorCitizen":    np.ones(n, dtype=int),
        "Partner":          np.zeros(n, dtype=int),
        "Dependents":       np.zeros(n, dtype=int),
        "tenure":           rng.integers(1, 6, n),
        "PhoneService":     np.ones(n, dtype=int),
        "MultipleLines":    np.ones(n, dtype=int),
        "InternetService":  np.ones(n, dtype=int),
        "OnlineSecurity":   np.zeros(n, dtype=int),
        "OnlineBackup":     np.zeros(n, dtype=int),
        "DeviceProtection": np.zeros(n, dtype=int),
        "TechSupport":      np.zeros(n, dtype=int),
        "StreamingTV":      np.ones(n, dtype=int),
        "StreamingMovies":  np.ones(n, dtype=int),
        "Contract":         np.zeros(n, dtype=int),
        "PaperlessBilling": np.ones(n, dtype=int),
        "PaymentMethod":    rng.integers(0, 4, n),
        "MonthlyCharges":   rng.uniform(85, 115, n),
        "TotalCharges":     rng.uniform(85, 600, n),
    })


def generate_report(
    data_path: str,
    output_path: str,
    drift_threshold: float,
    current_sample_size: int,
    simulate_drift: bool,
) -> dict:
    """
    Compare reference (training) data against current (production-like)
    data and generate an HTML drift report. Returns a summary dict.
    """
    # ── Load reference data ──────────────────────────────────────────────────
    ref_data = pd.read_csv(data_path)
    ref_data["TotalCharges"] = pd.to_numeric(ref_data["TotalCharges"], errors="coerce")
    ref_data["TotalCharges"] = ref_data["TotalCharges"].fillna(ref_data["TotalCharges"].median())

    # Encode categoricals the same way training does, so column dtypes match
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cat_cols = ref_data.select_dtypes(include="object").columns.tolist()
    for col in ["customerID", "Churn"]:
        if col in cat_cols:
            cat_cols.remove(col)
    for col in cat_cols:
        ref_data[col] = le.fit_transform(ref_data[col].astype(str))

    ref_features = ref_data[FEATURE_COLUMNS]

    # ── Build "current" data ─────────────────────────────────────────────────
    current_features = ref_features.sample(
        n=min(current_sample_size, len(ref_features)), random_state=42
    ).reset_index(drop=True)

    if simulate_drift:
        drift_batch = build_drift_batch(n=200)
        current_features = pd.concat(
            [current_features, drift_batch], ignore_index=True
        )
        print(f"[monitoring] Injected {len(drift_batch)} simulated drift rows")

    # ── Generate report ──────────────────────────────────────────────────────
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_features, current_data=current_features)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    print(f"[monitoring] Report saved: {output_path}")

    # ── Extract drift summary ────────────────────────────────────────────────
    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    n_drifted = drift_metric.get("number_of_drifted_columns", 0)
    n_total   = drift_metric.get("number_of_columns", len(FEATURE_COLUMNS))
    drift_share = drift_metric.get("share_of_drifted_columns", n_drifted / max(n_total, 1))

    summary = {
        "drifted_columns": n_drifted,
        "total_columns":   n_total,
        "drift_share":      round(float(drift_share), 4),
        "threshold":        drift_threshold,
        "alert":            bool(drift_share > drift_threshold),
    }

    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[monitoring] Drifted features: {n_drifted} / {n_total}")
    print(f"[monitoring] Drift share:      {drift_share:.2%}")

    if summary["alert"]:
        print(f"[monitoring] ALERT: Drift share {drift_share:.2%} exceeds threshold {drift_threshold:.2%}")
        print("[monitoring] ACTION REQUIRED: Consider retraining the model.")
    else:
        print(f"[monitoring] OK: Drift share {drift_share:.2%} is within acceptable range.")

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Generate an Evidently drift report")
    parser.add_argument("--data_path",   default="data/raw/telco_churn.csv")
    parser.add_argument("--output_path", default="monitoring/reports/drift_report.html")
    parser.add_argument("--threshold",   type=float, default=0.30)
    parser.add_argument("--sample_size", type=int,   default=500)
    parser.add_argument("--simulate_drift", action="store_true",
                        help="Inject a simulated new customer segment to force drift")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = generate_report(
        data_path=args.data_path,
        output_path=args.output_path,
        drift_threshold=args.threshold,
        current_sample_size=args.sample_size,
        simulate_drift=args.simulate_drift,
    )
    sys.exit(1 if summary["alert"] else 0)
