"""
ui/streamlit_app.py — Bonus Demo UI for the Churn Prediction API
===================================================================
This is an OPTIONAL, BONUS addition to the workshop — not a numbered
session. It sits on top of Session 5's FastAPI service and calls it over
plain HTTP, exactly like curl or Swagger UI does. The UI never loads the
model itself — this keeps the API as the single source of truth.

RUN LOCALLY (talks to the API on localhost:8000):
    streamlit run ui/streamlit_app.py

RUN AGAINST A DEPLOYED API (e.g. your Session 7 cloud URL):
    API_BASE_URL=http://YOUR_EC2_IP:8000 streamlit run ui/streamlit_app.py
    (or just paste the URL into the sidebar field once the app is open)

GOOD FOR:
    - Capstone presentations (Session 11) — much more visual than curl
    - Interview demos — click a button instead of typing JSON
    - Showing the project to non-technical people (TPO, faculty, family)
"""

import os
import sys

import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_maps import (
    CONTRACT_MAP,
    GENDER_MAP,
    INTERNET_DEPENDENT_MAP,
    INTERNET_SERVICE_MAP,
    MULTIPLE_LINES_MAP,
    PAYMENT_METHOD_MAP,
    YES_NO_MAP,
)

st.set_page_config(page_title="Churn Prediction — Live Demo", page_icon="📉", layout="wide")

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def keys(mapping: dict) -> list:
    return list(mapping.keys())


def index_of(mapping: dict, value: str) -> int:
    return keys(mapping).index(value)


# ── Preset example customers (for one-click demos) ───────────────────────────
PRESETS = {
    "custom": dict(
        gender="Male", senior=False, partner="No", dependents="No", tenure=24,
        phone="Yes", multiple="No", internet="Fiber optic", security="No",
        backup="Yes", protection="No", support="No", tv="Yes", movies="No",
        contract="Month-to-month", paperless="Yes", payment="Electronic check",
        monthly=65.5, total=1572.0,
    ),
    "low": dict(
        gender="Male", senior=False, partner="Yes", dependents="Yes", tenure=60,
        phone="Yes", multiple="Yes", internet="DSL", security="Yes",
        backup="Yes", protection="Yes", support="Yes", tv="Yes", movies="Yes",
        contract="Two year", paperless="No", payment="Bank transfer (automatic)",
        monthly=45.0, total=2700.0,
    ),
    "high": dict(
        gender="Female", senior=True, partner="No", dependents="No", tenure=2,
        phone="Yes", multiple="Yes", internet="Fiber optic", security="No",
        backup="No", protection="No", support="No", tv="Yes", movies="Yes",
        contract="Month-to-month", paperless="Yes", payment="Electronic check",
        monthly=95.5, total=191.0,
    ),
}

if "preset" not in st.session_state:
    st.session_state.preset = "custom"


# ── Sidebar: API connection ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API Connection")
    api_url = st.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
        help="Point this at localhost:8000 (local), your Docker container, "
             "or your Session 7 cloud URL (AWS EC2 / Render.com).",
    ).rstrip("/")

    if st.button("🔍 Check API Health", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("model_loaded"):
                st.success(
                    f"**{data['status'].upper()}**\n\n"
                    f"Model: `{data['model_name']}` v{data['model_version']}\n\n"
                    f"Source: {data['model_source']}"
                )
            else:
                st.warning(f"**{data['status'].upper()}** — model not loaded yet")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not reach `{api_url}`.\nIs the API running?")
        except Exception as e:
            st.error(f"❌ {e}")

    st.divider()
    st.caption(
        "Built during the **Krysha Academy MLOps Workshop**. "
        "This UI is a thin HTTP client — it calls the same `/predict` "
        "endpoint you tested with curl and Swagger UI in Session 5."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.caption("A simple demo UI in front of the FastAPI model-serving API (Session 5).")

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("📋 Load Low-Risk Example", use_container_width=True):
        st.session_state.preset = "low"
        st.rerun()
with col_b:
    if st.button("🔥 Load High-Risk Example", use_container_width=True):
        st.session_state.preset = "high"
        st.rerun()
with col_c:
    if st.button("↩️ Reset to Defaults", use_container_width=True):
        st.session_state.preset = "custom"
        st.rerun()

p = PRESETS[st.session_state.preset]
st.divider()


# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Customer Profile")
c1, c2, c3, c4 = st.columns(4)
with c1:
    gender = st.selectbox("Gender", keys(GENDER_MAP), index=index_of(GENDER_MAP, p["gender"]))
with c2:
    senior = st.checkbox("Senior Citizen", value=p["senior"])
with c3:
    partner = st.selectbox("Has Partner", keys(YES_NO_MAP), index=index_of(YES_NO_MAP, p["partner"]))
with c4:
    dependents = st.selectbox("Has Dependents", keys(YES_NO_MAP), index=index_of(YES_NO_MAP, p["dependents"]))

tenure = st.slider("Tenure (months as customer)", 0, 72, value=p["tenure"])

st.subheader("Services")
c1, c2, c3 = st.columns(3)
with c1:
    phone = st.selectbox("Phone Service", keys(YES_NO_MAP), index=index_of(YES_NO_MAP, p["phone"]))
    multiple = st.selectbox("Multiple Lines", keys(MULTIPLE_LINES_MAP), index=index_of(MULTIPLE_LINES_MAP, p["multiple"]))
    internet = st.selectbox("Internet Service", keys(INTERNET_SERVICE_MAP), index=index_of(INTERNET_SERVICE_MAP, p["internet"]))
with c2:
    security = st.selectbox("Online Security", keys(INTERNET_DEPENDENT_MAP), index=index_of(INTERNET_DEPENDENT_MAP, p["security"]))
    backup = st.selectbox("Online Backup", keys(INTERNET_DEPENDENT_MAP), index=index_of(INTERNET_DEPENDENT_MAP, p["backup"]))
    protection = st.selectbox("Device Protection", keys(INTERNET_DEPENDENT_MAP), index=index_of(INTERNET_DEPENDENT_MAP, p["protection"]))
with c3:
    support = st.selectbox("Tech Support", keys(INTERNET_DEPENDENT_MAP), index=index_of(INTERNET_DEPENDENT_MAP, p["support"]))
    tv = st.selectbox("Streaming TV", keys(INTERNET_DEPENDENT_MAP), index=index_of(INTERNET_DEPENDENT_MAP, p["tv"]))
    movies = st.selectbox("Streaming Movies", keys(INTERNET_DEPENDENT_MAP), index=index_of(INTERNET_DEPENDENT_MAP, p["movies"]))

st.subheader("Billing")
c1, c2, c3 = st.columns(3)
with c1:
    contract = st.selectbox("Contract", keys(CONTRACT_MAP), index=index_of(CONTRACT_MAP, p["contract"]))
    paperless = st.selectbox("Paperless Billing", keys(YES_NO_MAP), index=index_of(YES_NO_MAP, p["paperless"]))
with c2:
    payment = st.selectbox("Payment Method", keys(PAYMENT_METHOD_MAP), index=index_of(PAYMENT_METHOD_MAP, p["payment"]))
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=float(p["monthly"]), step=0.5)
with c3:
    suggested_total = round(tenure * monthly, 2)
    total = st.number_input(
        "Total Charges ($)", min_value=0.0, value=float(p["total"]), step=1.0,
        help=f"Suggested (tenure × monthly charges): ${suggested_total:,.2f}",
    )

st.divider()


# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    payload = {
        "gender": GENDER_MAP[gender],
        "SeniorCitizen": int(senior),
        "Partner": YES_NO_MAP[partner],
        "Dependents": YES_NO_MAP[dependents],
        "tenure": tenure,
        "PhoneService": YES_NO_MAP[phone],
        "MultipleLines": MULTIPLE_LINES_MAP[multiple],
        "InternetService": INTERNET_SERVICE_MAP[internet],
        "OnlineSecurity": INTERNET_DEPENDENT_MAP[security],
        "OnlineBackup": INTERNET_DEPENDENT_MAP[backup],
        "DeviceProtection": INTERNET_DEPENDENT_MAP[protection],
        "TechSupport": INTERNET_DEPENDENT_MAP[support],
        "StreamingTV": INTERNET_DEPENDENT_MAP[tv],
        "StreamingMovies": INTERNET_DEPENDENT_MAP[movies],
        "Contract": CONTRACT_MAP[contract],
        "PaperlessBilling": YES_NO_MAP[paperless],
        "PaymentMethod": PAYMENT_METHOD_MAP[payment],
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        with st.spinner("Calling the model API..."):
            r = requests.post(f"{api_url}/predict", json=payload, timeout=10)

        if r.status_code == 200:
            result = r.json()
            prob = result["churn_probability"]
            pred = result["prediction"]
            conf = result["confidence"]
            model_v = result["model_version"]

            st.divider()
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Churn Probability", f"{prob:.1%}")
            rc2.metric("Prediction", pred)
            rc3.metric("Confidence", conf)
            st.progress(min(max(prob, 0.0), 1.0))

            if pred == "Will churn":
                st.error(f"⚠️ High churn risk — probability {prob:.1%}. Consider a retention offer.")
            else:
                st.success(f"✅ Customer likely to stay — churn probability only {prob:.1%}.")

            st.caption(f"Served by model version: {model_v}")

            with st.expander("View raw API response"):
                st.json(result)
            with st.expander("View request payload sent to the API"):
                st.json(payload)
            with st.expander("Equivalent curl command"):
                import json as _json
                st.code(
                    f"curl -X POST {api_url}/predict \\\n"
                    f"  -H 'Content-Type: application/json' \\\n"
                    f"  -d '{_json.dumps(payload)}'",
                    language="bash",
                )

        elif r.status_code == 422:
            st.error("The API rejected the request (422 — validation error).")
            st.json(r.json())
        elif r.status_code == 503:
            st.warning("The model is not loaded on the API side yet (503). Check the sidebar health check.")
        else:
            st.error(f"Unexpected response: HTTP {r.status_code}")
            st.text(r.text)

    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Could not connect to the API at `{api_url}`.\n\n"
            f"Make sure the FastAPI service is running:\n\n"
            f"`uvicorn api.main:app --reload --port 8000`"
        )
    except requests.exceptions.Timeout:
        st.error("❌ The API took too long to respond (timeout after 10s).")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
