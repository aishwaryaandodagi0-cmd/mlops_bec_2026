"""
scripts/register_model.py — Register and Promote a Model (Session 4)
=======================================================================
USAGE:
    python scripts/register_model.py --run_id YOUR_BEST_RUN_ID

WHAT THIS DOES:
    1. Registers the model from an MLflow run to the Model Registry
    2. Adds a description and a validation tag
    3. Transitions the new version to "Staging"
    4. Promotes it to "Production" (archiving any previous Production version)

AFTER RUNNING:
    Your FastAPI app (Session 5) will load this exact model via:
        models:/churn-prediction-xgboost/Production
    No file paths. No guessing which pickle file is live.
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "churn-prediction-xgboost"


def register_and_promote(run_id: str, description: str = None) -> str:
    """
    Register a model from a run, then transition it through
    Staging → Production. Returns the new version number.
    """
    client = MlflowClient()

    # ── Step 1: Register ─────────────────────────────────────────────────────
    print(f"[register] Registering model from run: {run_id}")
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = model_version.version
    print(f"[register] Registered as version: {version}")

    # ── Step 2: Add description and tag ──────────────────────────────────────
    desc = description or "XGBoost churn prediction model. Validated by evaluate.py."
    client.update_model_version(
        name=MODEL_NAME, version=version, description=desc,
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=version, key="validated_by", value="evaluate.py",
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=version, key="source_run_id", value=run_id,
    )

    # ── Step 3: Transition to Staging ────────────────────────────────────────
    print("[register] Transitioning to Staging...")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    print(f"[register] Model version {version} is now in Staging")

    # ── Step 4: Promote to Production ────────────────────────────────────────
    print("[register] Promoting to Production...")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Production",
        archive_existing_versions=True,   # auto-archives the previous Production version
    )
    print(f"[register] Model version {version} is now in Production")
    print("[register] Previous Production version (if any) archived automatically")

    print("\n" + "═" * 55)
    print("  READY FOR SESSION 5")
    print("═" * 55)
    print(f"  Load this model in code with:")
    print(f"  >>> models:/{MODEL_NAME}/Production")
    print("═" * 55)

    return version


def parse_args():
    parser = argparse.ArgumentParser(description="Register and promote a model to Production")
    parser.add_argument("--run_id", required=True, help="MLflow Run ID to register")
    parser.add_argument("--description", default=None)
    parser.add_argument("--tracking_uri", default="http://98.130.129.135:5001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    register_and_promote(args.run_id, args.description)
