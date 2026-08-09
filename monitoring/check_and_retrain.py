"""
monitoring/check_and_retrain.py — Automated Retraining Trigger (Session 9)
============================================================================
USAGE:
    python monitoring/check_and_retrain.py

WHAT THIS DOES:
    1. Runs the drift report (generate_report.py) as a subprocess
    2. If the drift threshold was exceeded (exit code 1), automatically
       triggers `dvc repro` to retrain the pipeline
    3. This closes the MLOps flywheel: Monitor → Retrain → Track → ...

WIRE THIS INTO A SCHEDULER (production pattern, not required for workshop):
    - A cron job or GitHub Actions scheduled workflow could call this daily
    - On drift detection: retrains, then evaluate.py gates the new model
      before it can reach the Registry — the same safety net from Session 4
"""

import subprocess
import sys


def check_drift_and_retrain(simulate_drift: bool = False) -> int:
    """
    Run the drift report and trigger retraining if drift exceeds threshold.
    Returns 0 on success (no drift, or successful retrain), 1 on failure.
    """
    print("[check_and_retrain] Running drift detection...")

    cmd = ["python", "monitoring/generate_report.py"]
    if simulate_drift:
        cmd.append("--simulate_drift")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode == 1:
        # Drift threshold exceeded — trigger retraining
        print("[check_and_retrain] Drift detected. Triggering retraining pipeline (dvc repro)...")
        retrain = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
        print(retrain.stdout)

        if retrain.returncode == 0:
            print("[check_and_retrain] Retraining complete.")
            print("[check_and_retrain] Run `python src/evaluate.py` to check quality gates")
            print("[check_and_retrain] before promoting the new model to Production.")
            return 0
        else:
            print("[check_and_retrain] Retraining FAILED:")
            print(retrain.stderr, file=sys.stderr)
            return 1

    elif result.returncode == 0:
        print("[check_and_retrain] No significant drift detected. Retraining not required.")
        return 0

    else:
        print(f"[check_and_retrain] Unexpected exit code from drift check: {result.returncode}")
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_drift", action="store_true")
    args = parser.parse_args()

    exit_code = check_drift_and_retrain(simulate_drift=args.simulate_drift)
    sys.exit(exit_code)
