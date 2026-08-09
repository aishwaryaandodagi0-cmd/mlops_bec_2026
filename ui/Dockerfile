# ══════════════════════════════════════════════════════════════════════
#  ui/Dockerfile — Bonus Demo UI Container
#
#  IMPORTANT: build this from the PROJECT ROOT (not from inside ui/),
#  because it needs access to src/feature_maps.py which lives outside
#  the ui/ folder. Docker cannot COPY files from outside its build context.
#
#  BUILD (run this from the mlops-starter/ project root):
#    docker build -t churn-ui:1.0 -f ui/Dockerfile .
#
#  RUN:
#    docker run -p 8501:8501 -e API_BASE_URL=http://host.docker.internal:8000 churn-ui:1.0
#
#  Notice how small this image is compared to the API's Dockerfile —
#  no XGBoost, no MLflow, no scikit-learn. The UI is a thin HTTP client.
# ══════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

WORKDIR /app

COPY ui/requirements.txt ./ui/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt

# Preserve the SAME relative structure as the source repo (ui/ and src/
# as siblings under /app). streamlit_app.py walks up one directory to
# find src/feature_maps.py — this only works if the layout matches
# exactly what the app expects locally. Only feature_maps.py is copied
# from src/ (it has zero ML dependencies) — no need for the heavier
# training/serving modules inside this lightweight UI image.
COPY ui/streamlit_app.py ./ui/streamlit_app.py
COPY src/feature_maps.py ./src/feature_maps.py

EXPOSE 8501

ENV API_BASE_URL=http://host.docker.internal:8000

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
