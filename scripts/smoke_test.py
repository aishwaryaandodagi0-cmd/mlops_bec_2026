"""
scripts/smoke_test.py — Pre-Workshop Environment Verification
================================================================
USAGE:
    python scripts/smoke_test.py

WHAT THIS DOES:
    Checks every tool used across all 3 workshop days in one go.
    Prints a green [OK] for each passing check, or a red [FAIL] with
    a hint for anything broken. Run this BEFORE Day 1 — see the
    Pre-Workshop Setup Guide, Step 14.
"""

import importlib
import os
import shutil
import subprocess
import sys

OK   = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

checks_passed = 0
checks_failed = 0


def check(label: str, condition: bool, hint: str = ""):
    global checks_passed, checks_failed
    if condition:
        print(f"  {OK}  {label}")
        checks_passed += 1
    else:
        print(f"  {FAIL}  {label}")
        if hint:
            print(f"         → {hint}")
        checks_failed += 1


def check_import(label: str, module_name: str, hint: str = ""):
    try:
        importlib.import_module(module_name)
        check(label, True)
    except ImportError:
        check(label, False, hint or f"pip install {module_name}")


def check_command(label: str, command: list, hint: str = ""):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        check(label, result.returncode == 0, hint)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        check(label, False, hint)


def main():
    print("\n" + "═" * 60)
    print("  MLOPS WORKSHOP — ENVIRONMENT SMOKE TEST")
    print("═" * 60 + "\n")

    print("── Python & Core Tools ─────────────────────────────────")
    check("Python version >= 3.10",
          sys.version_info >= (3, 10),
          "Install Python 3.10 or 3.11")
    check_command("Git installed", ["git", "--version"], "Install Git")
    check_command("Docker installed", ["docker", "--version"], "Install Docker Desktop")
    check_command("Docker daemon running", ["docker", "info"],
                  "Open the Docker Desktop application")

    print("\n── Python Packages ──────────────────────────────────────")
    check_import("pandas",     "pandas")
    check_import("numpy",      "numpy")
    check_import("scikit-learn", "sklearn")
    check_import("XGBoost",    "xgboost")
    check_import("MLflow",     "mlflow")
    check_import("DVC",        "dvc")
    check_import("FastAPI",    "fastapi")
    check_import("Uvicorn",    "uvicorn")
    check_import("Pydantic",   "pydantic")
    check_import("Evidently",  "evidently")
    check_import("pytest",     "pytest")
    check_import("httpx",      "httpx")

    print("\n── AWS & Cloud Tools ────────────────────────────────────")
    check_command("AWS CLI installed", ["aws", "--version"],
                  "Install: https://aws.amazon.com/cli/")
    check_import("boto3 (AWS SDK)", "boto3")

    print("\n── Datasets ──────────────────────────────────────────────")
    check("Telco Churn dataset present",
          os.path.exists("data/raw/telco_churn.csv"),
          "Download from Kaggle: Telco Customer Churn, place in data/raw/")
    check("Loan Default dataset present",
          os.path.exists("data/raw/loan_default.csv"),
          "Download from Kaggle: Give Me Some Credit, place in data/raw/")

    print("\n── Project Structure ────────────────────────────────────")
    for path in ["src/train.py", "api/main.py", "tests/test_api.py",
                 "monitoring/generate_report.py", "Dockerfile", "dvc.yaml",
                 "requirements.txt", ".github/workflows/ci.yml"]:
        check(f"{path} exists", os.path.exists(path))

    print("\n" + "═" * 60)
    total = checks_passed + checks_failed
    print(f"  RESULT: {checks_passed}/{total} checks passed")
    print("═" * 60)

    if checks_failed == 0:
        print("\n  🎉 All checks passed! You are ready for Day 1.\n")
        sys.exit(0)
    else:
        print(f"\n  ⚠️  {checks_failed} check(s) failed. Fix the items above")
        print("     before Day 1, or ask in the workshop support group.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
