# MLOps Starter — Customer Churn Prediction

> Built during the **Krysha Academy MLOps Workshop** — a 3-day, hands-on journey
> from a broken Jupyter notebook to a live, monitored, cloud-deployed ML system.

## What This Project Does

Predicts customer churn probability for a telecom company using an XGBoost
classifier, fully productionised with industry-standard MLOps tooling.

## The 7-Layer Pipeline

| Layer | Purpose | Tool |
|---|---|---|
| 1 | Experiment Tracking | MLflow |
| 2 | Data Versioning | DVC |
| 3 | Model Registry | MLflow Model Registry |
| 4 | Model Serving API | FastAPI |
| 5 | Containerisation | Docker |
| 6 | Cloud Deployment | AWS EC2 / Render.com |
| 7 | CI/CD + Monitoring | GitHub Actions + Evidently AI |

## Live Demo

- **API URL**: _add your deployed URL here after Session 7_
- **Swagger UI**: `<API_URL>/docs`
- **Health check**: `<API_URL>/health`

```bash
curl -X POST <API_URL>/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 24, "PhoneService": 1, "MultipleLines": 0,
    "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 1,
    "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 1,
    "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
    "PaymentMethod": 2, "MonthlyCharges": 65.5, "TotalCharges": 1572.0
  }'
```

## Project Structure

```
mlops-starter/
├── data/
│   ├── raw/              # Original datasets — never modify
│   └── processed/        # Feature-engineered datasets
├── src/
│   ├── train.py          # MLflow-tracked training (Session 2)
│   ├── evaluate.py        # Automated quality gates (Session 4)
│   ├── predict.py        # Shared prediction logic (Session 5)
│   └── feature_maps.py   # Shared label-encoding maps (used by bonus UI)
├── api/
│   └── main.py            # FastAPI app (Session 5)
├── ui/                     # BONUS: optional Streamlit demo UI
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   └── test_api.py       # Unit tests (Session 8)
├── monitoring/
│   ├── generate_report.py     # Evidently drift report (Session 9)
│   └── check_and_retrain.py   # Automated retraining trigger (Session 9)
├── scripts/
│   ├── register_model.py  # Register + promote to Production (Session 4)
│   ├── find_best_run.py   # Find best MLflow run by metric
│   └── smoke_test.py      # Pre-workshop environment check
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline (Session 8)
├── Dockerfile              # Container definition (Session 6)
├── docker-compose.yml       # BONUS: run API + UI together
├── dvc.yaml                # Reproducible pipeline (Session 3)
├── params.yaml             # Versioned hyperparameters
├── requirements.txt
└── README.md
```

## Bonus: Demo UI (Streamlit)

An optional Streamlit UI sits in front of the FastAPI service — useful for
capstone presentations (Session 11), interview demos, and showing the
project to non-technical people. It is a thin HTTP client: it never loads
the model itself, it just calls `/predict` and `/health` exactly like curl
or Swagger UI does.

```bash
# Run locally (talks to the API on localhost:8000)
uvicorn api.main:app --reload --port 8000     # terminal 1
streamlit run ui/streamlit_app.py             # terminal 2
# open http://localhost:8501
```

**Or run everything together with Docker Compose:**

```bash
docker compose up --build
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

Point the UI at any API — local, Docker, or your Session 7 cloud
deployment — by changing the "API Base URL" field in the sidebar, or:

```bash
API_BASE_URL=http://YOUR_EC2_IP:8000 streamlit run ui/streamlit_app.py
```

Files: `ui/streamlit_app.py`, `ui/requirements.txt`, `ui/Dockerfile`,
`src/feature_maps.py` (shared label-encoding maps), `docker-compose.yml`.

## Quickstart

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd mlops-starter
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Start MLflow tracking server
mlflow ui --port 5001
# open http://localhost:5001

# 3. Train the model
python src/train.py

# 4. Evaluate against quality gates
python src/evaluate.py --run_id <RUN_ID_FROM_STEP_3>

# 5. Register to Production
python scripts/register_model.py --run_id <RUN_ID_FROM_STEP_3>

# 6. Run the API
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000/docs

# 7. Run tests
pytest tests/ -v

# 8. Build and run with Docker
docker build -t churn-api:1.0 .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 churn-api:1.0

# 9. Generate a monitoring report
python monitoring/generate_report.py --simulate_drift
open monitoring/reports/drift_report.html

# 10. (Bonus) Launch the demo UI
streamlit run ui/streamlit_app.py
# open http://localhost:8501
```

## Tech Stack

Python 3.11 · XGBoost · MLflow · DVC · FastAPI · Streamlit (bonus UI) ·
Docker · GitHub Actions · AWS EC2 / Render.com · Evidently AI · pytest

## Data Sources

- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — used in the Day 3 capstone

## Author

Built as part of the Krysha Academy MLOps Workshop.
