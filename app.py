import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import json
from datetime import datetime


# ============================================================
# AGRIWISE - SMART AGRICULTURE PREDICTION SYSTEM USING ML
# ============================================================

st.set_page_config(
    page_title="AgriWise | Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# AGRIWISE THEME SETTINGS
# ============================================================

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "🌙 Dark"

theme_space, theme_control = st.columns([8, 2])

with theme_control:
    st.segmented_control(
        "Theme",
        options=["☀️ Light", "🌙 Dark"],
        key="theme_choice",
        label_visibility="collapsed"
    )

st.session_state.dark_mode = (
    st.session_state.theme_choice == "🌙 Dark"
)

BASE_DIR = Path(__file__).resolve().parent
LEAFGUARD_MODEL_PATH = BASE_DIR / "models" / "leafguard" / "leafguard_best_model.pth"


# ============================================================
# ERROR LOGGING
# ============================================================

LOG_FILE = BASE_DIR / "agriwise_errors.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True
)

logger = logging.getLogger("AgriWise")


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 10%, #123d32 0%, transparent 45%),
            radial-gradient(ellipse at 90% 80%, #174b36 0%, transparent 40%),
            linear-gradient(135deg, #071713, #0b211b 55%, #102a20);
        color: #edf7ef;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091c17, #102c21);
        border-right: 1px solid #28523d;
    }

    [data-testid="stSidebar"] * {
        color: #e5f5e9;
    }

    .hero {
        padding: 35px;
        border: 1px solid #315b43;
        border-radius: 22px;
        background: linear-gradient(120deg, #123c2d, #10251e);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        color: #a7f3b4;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d1e9d7;
        font-size: 16px;
    }

    .module-card {
    background: linear-gradient(135deg, #164b35, #103c2b);
    border: 1px solid #397653;
    border-radius: 18px;
    padding: 25px;
    min-height: 170px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
    transition: all 0.3s ease;
}

.module-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 12px 25px rgba(0, 0, 0, 0.28);
    border-color: #75b58a;
}

    .module-card h3 {
        color: #a7f3b4;
        margin-bottom: 10px;
    }

    .module-card p {
        color: #d2e6d8;
        font-size: 14px;
    }

    .result-card {
        background: #123c2b;
        border: 1px solid #4b9861;
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
    }

    .result-card h2 {
        color: #a7f3b4;
    }

    .small-note {
        color: #a9c6b2;
        font-size: 12px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #16834c, #20a45d);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 22px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #25b86b;
        color: white;
        border: none;
    }

    [data-testid="stMetric"] {
        background: rgba(25, 69, 48, 0.65);
        border: 1px solid #315b43;
        padding: 15px;
        border-radius: 14px;
    }

    h1, h2, h3 {
        color: #e7f8eb;
    }

    .stCaption {
        color: #b4cdbb;
    }

    hr {
        border-color: #315b43;
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 12px 0 22px;
    }

    .dashboard-card {
        padding: 22px;
        border-radius: 17px;
        background: rgba(22, 57, 42, 0.78);
        border: 1px solid #315b43;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-4px);
        border-color: #75b58a;
    }

    .dashboard-icon {
        font-size: 22px;
        margin-bottom: 12px;
    }

    .dashboard-value {
        font-size: 28px;
        font-weight: 800;
        color: #a7f3b4;
    }

    .dashboard-label {
        font-size: 13px;
        margin-top: 5px;
        color: #d2e6d8;
    }

    @media (max-width: 650px) {
        .dashboard-grid {
            grid-template-columns: 1fr;
        }
    }

    
    /* SIDEBAR NAVIGATION */

    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 10px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(35, 85, 57, 0.25);
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 12px 14px;
        transition: all 0.25s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(57, 130, 83, 0.3);
        border-color: #548d68;
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #176b43, #245f3e);
        border: 1px solid #65a77a;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-weight: 600;
        font-size: 14px;
    }


</style>
""", unsafe_allow_html=True)

# ============================================================
# LIGHT MODE DESIGN
# ============================================================

if not st.session_state.dark_mode:

    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #f5f8f2,
                #eaf2e7,
                #f8f7ef
            ) !important;
            color: #243c2e !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #e2eee2,
                #d5e6d7
            ) !important;
            border-right: 1px solid #bfd4c1;
        }

        [data-testid="stSidebar"] * {
            color: #244b35 !important;
        }

        .hero {
            background: linear-gradient(
                120deg,
                #dcebdc,
                #c9e2ce
            ) !important;
            border-color: #b5d0b8;
        }

        .hero h1,
        .module-card h3 {
            color: #24633d !important;
        }

        .hero p,
        .module-card p {
            color: #405c49 !important;
        }

        .module-card {
            background: rgba(255,255,255,0.85) !important;
            border-color: #cbdccb !important;
        }

        .result-card {
            background: #e2f0e3 !important;
            border-color: #a8cbaa !important;
        }

        .result-card h2 {
            color: #24633d !important;
        }

        [data-testid="stMetric"] {
            background: #ffffff !important;
            border-color: #cbdccb !important;
        }

        h1, h2, h3, p, label {
            color: #263e30;
        }

        hr {
            border-color: #cbdccb !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATHS = {
    "CropWise": "models/cropwise/best_cropwise_model.joblib",
    "SoilSense": "models/soilsense/best_soilsense_model.joblib",
    "FertilizerWise": "models/fertilizerwise/best_fertilizerwise_model.joblib",
    "YieldWise": "models/yieldwise/best_yieldwise_model.joblib",
    "RainWise": "models/rainwise/rainwise_best_model.pkl",
}

DATA_PATHS = {
    "CropWise": "data/processed/cropwise/cropwise_train.csv",
    "SoilSense": "data/processed/soilsense/soilsense_train.csv",
    "FertilizerWise": "data/processed/fertilizerwise/fertilizerwise_train.csv",
    "YieldWise": "data/processed/yieldwise/yieldwise_train.csv",
}


# ============================================================
# LOAD MODELS AND DATASETS
# ============================================================

@st.cache_resource
def load_model(module):
    path = BASE_DIR / MODEL_PATHS[module]

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


@st.cache_data
def load_dataset(module):
    path = BASE_DIR / DATA_PATHS[module]

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path)

@st.cache_resource
def load_leafguard_model():

    if not LEAFGUARD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LeafGuard model not found: {LEAFGUARD_MODEL_PATH}"
        )

    checkpoint = torch.load(
        LEAFGUARD_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def load_module(module):
    try:
        model = load_model(module)
        data = load_dataset(module)
        return model, data

    except Exception as error:
        logger.exception("Unable to load module: %s", module)
        st.error(f"Unable to load {module}: {error}")
        return None, None


# ============================================================
# COMMON INPUT HELPERS
# ============================================================

def numeric_input(data, column, label, integer=False):
    minimum = float(data[column].min())
    maximum = float(data[column].max())
    median = float(data[column].median())

    if integer:
        minimum = int(np.floor(minimum))
        maximum = int(np.ceil(maximum))
        default = int(round(median))

        if minimum == maximum:
            maximum += 1

        return st.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=1
        )

    if minimum == maximum:
        maximum = minimum + 1.0

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=float(median),
        step=0.1,
        format="%.2f"
    )


def categorical_input(data, column, label, preferred=None):
    options = sorted(data[column].dropna().unique().tolist())

    if not options:
        raise ValueError(f"No options available for {column}")

    default_index = 0

    if preferred in options:
        default_index = options.index(preferred)

    return st.selectbox(
        label,
        options,
        index=default_index
    )


def show_probabilities(model, input_data):
    if not hasattr(model, "predict_proba"):
        return

    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    results = pd.DataFrame({
        "Class": classes,
        "Probability": probabilities
    })

    results = results.sort_values(
        "Probability",
        ascending=False
    ).head(5)

    st.subheader("Top prediction probabilities")

    for _, row in results.iterrows():
        percentage = float(row["Probability"]) * 100

        st.write(f"**{row['Class']}** — {percentage:.2f}%")

        st.progress(
            min(max(float(row["Probability"]), 0.0), 1.0)
        )

    st.caption(
        "Probabilities represent model outputs, not guaranteed correctness."
    )

def get_leafguard_transform(checkpoint):

    return transforms.Compose([
        transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"]
        )
    ])

def predict_leafguard(image, model, checkpoint):

    transform = get_leafguard_transform(checkpoint)

    image_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        top_probabilities, top_indices = torch.topk(
            probabilities, k=3
        )

    class_names = checkpoint["class_names"]

    predictions = []

    for probability, index in zip(
        top_probabilities.tolist(),
        top_indices.tolist()
    ):

        predictions.append({
            "class": class_names[index],
            "confidence": probability
        })

    return predictions


# ============================================================
# RECENT ACTIVITY TRACKER
# ============================================================

ACTIVITY_FILE = Path("data/recent_activities.json")


def save_activity(module, prediction):

    ACTIVITY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    activities = []

    if ACTIVITY_FILE.exists():
        try:
            with open(ACTIVITY_FILE, "r", encoding="utf-8") as file:
                activities = json.load(file)
        except (json.JSONDecodeError, OSError):
            activities = []

    activities.insert(0, {
        "module": module,
        "prediction": str(prediction),
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
    })

    activities = activities[:50]

    with open(ACTIVITY_FILE, "w", encoding="utf-8") as file:
        json.dump(activities, file, indent=4)


def load_activities():

    if not ACTIVITY_FILE.exists():
        return []

    try:
        with open(ACTIVITY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def show_module_chart(module):

    st.divider()
    st.subheader("📊 Prediction History")

    activities = [
        item for item in load_activities()
        if isinstance(item, dict)
        and item.get("module") == module
    ]

    if not activities:
        st.info(
            f"No prediction history available for {module} yet. "
            "Make a prediction to see this chart."
        )
        return

    # CROPWISE - Recommended crop distribution
    if module == "CropWise":
        labels = [
            str(item.get("prediction", "")).strip().title()
            for item in activities
            if str(item.get("prediction", "")).strip()
        ]

        if not labels:
            st.info("No valid crop predictions available.")
            return

        counts = (
            pd.Series(labels).value_counts()
            .rename_axis("Crop").reset_index(name="Count")
        )

        fig = px.pie(
            counts,
            names="Crop",
            values="Count",
            hole=0.55,
            title="Recommended Crop Distribution"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

    # SOILSENSE - Soil class frequency
    elif module == "SoilSense":
        labels = [
            str(item.get("prediction", "")).strip()
            for item in activities
            if str(item.get("prediction", "")).strip()
        ]

        if not labels:
            st.info("No valid soil predictions available.")
            return

        counts = (
            pd.Series(labels).value_counts().sort_index()
            .rename_axis("Soil Class").reset_index(name="Count")
        )

        fig = px.bar(
            counts,
            x="Soil Class",
            y="Count",
            text="Count",
            title="Soil Class Prediction Frequency"
        )
        fig.update_traces(textposition="outside")

    # FERTILIZERWISE - Fertilizer distribution
    elif module == "FertilizerWise":
        labels = [
            str(item.get("prediction", "")).strip()
            for item in activities
            if str(item.get("prediction", "")).strip()
        ]

        if not labels:
            st.info("No valid fertilizer predictions available.")
            return

        counts = (
            pd.Series(labels).value_counts()
            .rename_axis("Fertilizer").reset_index(name="Count")
        )

        fig = px.pie(
            counts,
            names="Fertilizer",
            values="Count",
            hole=0.55,
            title="Recommended Fertilizer Distribution"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")

    # YIELDWISE - Predicted yield history
    elif module == "YieldWise":
        chronological = list(reversed(activities))

        values = pd.to_numeric(
            pd.Series([item.get("prediction") for item in chronological]),
            errors="coerce"
        ).dropna().tolist()

        if not values:
            st.info("No valid yield predictions available.")
            return

        chart_data = pd.DataFrame({
            "Prediction Number": range(1, len(values) + 1),
            "Predicted Yield (hg/ha)": values
        })

        fig = px.scatter(
            chart_data,
            x="Prediction Number",
            y="Predicted Yield (hg/ha)",
            title="Predicted Crop Yield History"
        )

    # RAINWISE - Predicted rainfall history
    elif module == "RainWise":
        chronological = list(reversed(activities))

        values = pd.to_numeric(
            pd.Series([item.get("prediction") for item in chronological]),
            errors="coerce"
        ).dropna().tolist()

        if not values:
            st.info("No valid rainfall predictions available.")
            return

        chart_data = pd.DataFrame({
            "Prediction Number": range(1, len(values) + 1),
            "Predicted Rainfall (mm)": values
        })

        fig = px.area(
            chart_data,
            x="Prediction Number",
            y="Predicted Rainfall (mm)",
            title="Predicted Rainfall History"
        )

    # LEAFGUARD - Disease prediction frequency
    elif module == "LeafGuard":
        labels = []

        for item in activities:
            prediction = str(item.get("prediction", "")).strip()
            disease = prediction.split(" (", 1)[0].strip()

            if disease:
                labels.append(disease)

        if not labels:
            st.info("No valid disease predictions available.")
            return

        counts = (
            pd.Series(labels).value_counts()
            .rename_axis("Disease").reset_index(name="Count")
        )

        fig = px.bar(
            counts,
            x="Count",
            y="Disease",
            orientation="h",
            text="Count",
            title="Plant Disease Prediction Frequency"
        )
        fig.update_traces(textposition="outside")

    else:
        st.info("No chart is configured for this module.")
        return

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    


# ============================================================
# AGRIWISE DASHBOARD
# ============================================================

def dashboard_page():

    st.title("📊 Dashboard")

    # MODEL PERFORMANCE
    st.subheader("🎯 Model Performance")
    st.caption("Performance metrics of the six trained ML models.")

    # FIRST ROW
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🌾 CropWise")
            st.metric("Test Accuracy", "99.55%")
            st.caption("Selected Model: Random Forest")

    with col2:
        with st.container(border=True):
            st.markdown("### 🧪 SoilSense")
            st.metric("Test Accuracy", "88.07%")
            st.caption("Selected Model: Random Forest")

    with col3:
        with st.container(border=True):
            st.markdown("### 🌿 FertilizerWise")
            st.metric("Test Accuracy", "97.12%")
            st.caption("Selected Model: Decision Tree")

    # SECOND ROW
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 📈 YieldWise")
            st.metric("Test R² Score", "93.22%")
            st.caption("Selected Model: Random Forest")
            st.caption("MAE: 11,663.66")
            st.caption("RMSE: 22,747.64")

    with col2:
        with st.container(border=True):
            st.markdown("### 🌧️ RainWise")
            st.metric("Validation R² Score", "34.84%")
            st.caption("Selected Model: Random Forest")
            st.caption("MAE: 2.8580")
            st.caption("RMSE: 7.7378")

    with col3:
        with st.container(border=True):
            st.markdown("### 🍃 LeafGuard")
            st.metric("Best Validation Accuracy", "96.66%")
            st.caption("Best Epoch: 10")
            st.caption("Model: MobileNetV2")

    st.info(
        "Note: LeafGuard reports validation accuracy because "
        "its training workflow did not use a separate test dataset. "
        "R² scores are regression metrics, not classification accuracy."
    )

    st.divider()

    # RECENT ACTIVITIES
    st.subheader("🕒 Recent Activities")

    activities = load_activities()

    if not activities:

        st.info("No recent predictions available yet.")

    else:

        for activity in activities:

            with st.container(border=True):

                st.markdown(
                    f"**🌱 {activity['module']}**"
                )

                st.write(
                    f"Prediction: {activity['prediction']}"
                )

                st.caption(activity["time"])

    st.caption("AgriWise | Smart Agriculture Prediction System")




# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## 🌱 AgriWise")

st.sidebar.caption("Smart Agriculture Prediction System")

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🌾 CropWise",
        "🧪 SoilSense",
        "🌿 FertilizerWise",
        "📈 YieldWise",
        "🌧️ RainWise",
        "🍃 LeafGuard"
    ],
    key="main_navigation"
)

page = page.split(" ", 1)[1]


st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Powered Agriculture")
st.sidebar.caption("Six independent prediction modules")



# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # HERO SECTION
    st.markdown("""
    <div class="hero">
        <p style="font-size:14px; letter-spacing:2px; margin-bottom:8px;">
            SMART AGRICULTURE PLATFORM
        </p>
        <h1>🌱 AgriWise</h1>
        <p><b>Smart Agriculture Prediction System using ML</b></p>
        <p>
            Making agriculture smarter through data-driven insights.
            Explore intelligent tools for crop recommendation, soil
            analysis, fertilizer selection, yield prediction,
            rainfall estimation and plant disease detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

   
    # DASHBOARD OVERVIEW

    st.markdown("### 📊 AgriWise at a Glance")
    st.caption("An overview of the platform")

    st.markdown(
        """<div class="dashboard-grid"><div class="dashboard-card"><div class="dashboard-icon">▦</div><div class="dashboard-value">06</div><div class="dashboard-label">Prediction Modules</div></div><div class="dashboard-card"><div class="dashboard-icon">🧠</div><div class="dashboard-value">ML</div><div class="dashboard-label">Technology</div></div><div class="dashboard-card"><div class="dashboard-icon">🌱</div><div class="dashboard-value">Agri</div><div class="dashboard-label">Domain</div></div></div>""",
        unsafe_allow_html=True
    )

    st.markdown("---")


    # MODULE SECTION
    st.markdown("### Explore Our Modules")
    st.caption("Choose a module from the sidebar to get started.")

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("""
        <div class="module-card">
            <h3>🌾 CropWise</h3>
            <p>
                Recommends a suitable crop based on soil nutrients
                and environmental conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="module-card">
            <h3>🧪 SoilSense</h3>
            <p>
                Analyses soil nutrient values and predicts
                the corresponding soil fertility class.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="module-card">
            <h3>🌧️ RainWise</h3>
            <p>
                Predicts next-day rainfall using historical weather
                data and machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="module-card">
            <h3>🌿 FertilizerWise</h3>
            <p>
                Recommends a fertilizer based on nutrient values,
                soil colour, crop and district.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="module-card">
            <h3>📈 YieldWise</h3>
            <p>
                Predicts crop yield using historical agricultural
                and environmental information.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="module-card">
            <h3>🍃 LeafGuard</h3>
            <p>
                Identifies possible plant diseases from uploaded
                leaf images using deep learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ABOUT SECTION
    st.markdown("---")

    st.markdown("### About AgriWise")

    st.markdown("""
    <div class="module-card">
        <h3>🌱 Agriculture Meets Artificial Intelligence</h3>
        <p>
            AgriWise is a software-based agriculture prediction
            platform featuring six independently trained
            machine-learning models.
        </p>
        <p>
            Each module addresses a different agricultural task
            and generates predictions using user-provided inputs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Note: Predictions are model estimates based on the datasets "
        "used for training. They should not replace professional "
        "agricultural advice."
    )

    # FOOTER
    st.markdown("""
    <div style="
        text-align:center;
        padding:18px 5px 8px;
        color:#9aaa9e;
        font-size:13px;
    ">
        🌱 AgriWise · Smart Agriculture Prediction System
        <br>
        Powered by Machine Learning
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CROPWISE
# ============================================================

def cropwise_page():

    st.title("🌾 CropWise")
    st.write("Crop Recommendation System")

    st.caption(
        "Enter soil nutrient and environmental values to predict a suitable crop."
    )

    model, data = load_module("CropWise")

    

    if model is None:
        return

    with st.form("cropwise_form"):

        st.subheader("Soil Nutrients")

        c1, c2, c3 = st.columns(3)

        with c1:
            N = numeric_input(data, "N", "Nitrogen (N)", True)

        with c2:
            P = numeric_input(data, "P", "Phosphorus (P)", True)

        with c3:
            K = numeric_input(data, "K", "Potassium (K)", True)

        st.subheader("Environmental Conditions")

        c1, c2 = st.columns(2)

        with c1:
            temperature = numeric_input(
                data, "temperature", "Temperature (°C)"
            )

            humidity = numeric_input(
                data, "humidity", "Humidity (%)"
            )

        with c2:
            ph = numeric_input(data, "ph", "Soil pH")

            rainfall = numeric_input(
                data, "rainfall", "Rainfall (mm)"
            )

        submitted = st.form_submit_button(
            "Predict Suitable Crop",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }])

        try:
            prediction = model.predict(input_data)[0]
            save_activity("CropWise", prediction)

            st.markdown(f"""
            <div class="result-card">
                <h2>🌱 Recommended Crop</h2>
                <h1>{prediction.title()}</h1>
            </div>
            """, unsafe_allow_html=True)

            show_probabilities(model, input_data)

        except Exception as error:
            logger.exception("CropWise prediction failed")
            st.error(f"Prediction failed: {error}")

    show_module_chart("CropWise")


# ============================================================
# SOILSENSE
# ============================================================

def soilsense_page():

    st.title("🧪 SoilSense")
    st.write("Soil Fertility Analysis")

    st.caption(
        "Enter soil nutrient values to predict the soil fertility class."
    )

    model, data = load_module("SoilSense")

    if model is None:
        return

    with st.form("soilsense_form"):

        st.subheader("Primary Nutrients")

        c1, c2, c3 = st.columns(3)

        with c1:
            N = numeric_input(data, "N", "Nitrogen (N)", True)

        with c2:
            P = numeric_input(data, "P", "Phosphorus (P)")

        with c3:
            K = numeric_input(data, "K", "Potassium (K)", True)

        st.subheader("Soil Properties")

        c1, c2, c3 = st.columns(3)

        with c1:
            ph = numeric_input(data, "pH", "Soil pH")

            EC = numeric_input(data, "EC", "Electrical Conductivity")

            OC = numeric_input(data, "OC", "Organic Carbon")

            S = numeric_input(data, "S", "Sulphur")

        with c2:
            Zn = numeric_input(data, "Zn", "Zinc")

            Fe = numeric_input(data, "Fe", "Iron")

            Cu = numeric_input(data, "Cu", "Copper")

        with c3:
            Mn = numeric_input(data, "Mn", "Manganese")

            B = numeric_input(data, "B", "Boron")

        submitted = st.form_submit_button(
            "Analyse Soil",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "pH": ph,
            "EC": EC,
            "OC": OC,
            "S": S,
            "Zn": Zn,
            "Fe": Fe,
            "Cu": Cu,
            "Mn": Mn,
            "B": B
        }])

        try:
            prediction = model.predict(input_data)[0]
            save_activity("SoilSense", prediction)

            st.markdown(f"""
            <div class="result-card">
                <h2>🧪 Predicted Soil Class</h2>
                <h1>Class {prediction}</h1>
            </div>
            """, unsafe_allow_html=True)

            st.info(
                "The dataset uses numerical classes. "
                "Class meanings have not been independently verified."
            )

            show_probabilities(model, input_data)

        except Exception as error:
            logger.exception("SoilSense analysis failed")
            st.error(f"Soil analysis failed: {error}")

    show_module_chart("SoilSense")


# ============================================================
# FERTILIZERWISE
# ============================================================

def fertilizerwise_page():

    st.title("🌿 FertilizerWise")
    st.write("Fertilizer Recommendation System")

    st.caption(
        "Provide soil nutrients, crop and environmental information "
        "to predict a fertilizer."
    )

    model, data = load_module("FertilizerWise")

    if model is None:
        return

    st.warning(
        "Dataset coverage: five districts in Western Maharashtra. "
        "Predictions should not be assumed suitable for other regions."
    )

    with st.form("fertilizerwise_form"):

        st.subheader("Soil Nutrients")

        c1, c2, c3 = st.columns(3)

        with c1:
            nitrogen = numeric_input(
                data, "Nitrogen", "Nitrogen", True
            )

        with c2:
            phosphorus = numeric_input(
                data, "Phosphorus", "Phosphorus", True
            )

        with c3:
            potassium = numeric_input(
                data, "Potassium", "Potassium", True
            )

        st.subheader("Soil and Environmental Details")

        c1, c2 = st.columns(2)

        with c1:
            ph = numeric_input(data, "pH", "Soil pH")

            rainfall = numeric_input(
                data, "Rainfall", "Rainfall", True
            )

        with c2:
            temperature = numeric_input(
                data, "Temperature", "Temperature (°C)", True
            )

            district = categorical_input(
                data, "District_Name", "District"
            )

        st.subheader("Crop Information")

        c1, c2 = st.columns(2)

        with c1:
            soil_color = categorical_input(
                data, "Soil_color", "Soil Colour"
            )

        with c2:
            crop = categorical_input(
                data, "Crop", "Crop"
            )

        submitted = st.form_submit_button(
            "Recommend Fertilizer",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "pH": ph,
            "Rainfall": rainfall,
            "Temperature": temperature,
            "District_Name": district,
            "Soil_color": soil_color,
            "Crop": crop
        }])

        try:
            prediction = model.predict(input_data)[0]
            save_activity("FertilizerWise", prediction)

            st.markdown(f"""
            <div class="result-card">
                <h2>🌿 Recommended Fertilizer</h2>
                <h1>{prediction}</h1>
            </div>
            """, unsafe_allow_html=True)

            show_probabilities(model, input_data)

        except Exception as error:
            logger.exception("FertilizerWise prediction failed")
            st.error(f"Fertilizer prediction failed: {error}")

    show_module_chart("FertilizerWise")


# ============================================================
# YIELDWISE
# ============================================================

def yieldwise_page():

    st.title("📈 YieldWise")
    st.write("Crop Yield Prediction")

    st.caption(
        "Enter historical agricultural and environmental information "
        "to estimate crop yield."
    )

    model, data = load_module("YieldWise")

    if model is None:
        return

    st.info(
        "This module uses an international historical dataset. "
        "Its predictions are not specific to Tamil Nadu."
    )

    with st.form("yieldwise_form"):

        st.subheader("Crop Information")

        c1, c2 = st.columns(2)

        with c1:
            area = categorical_input(
                data,
                "Area",
                "Country / Area",
                preferred="India"
            )

        with c2:
            crop = categorical_input(
                data,
                "Item",
                "Crop",
                preferred="Maize"
            )

        year = numeric_input(
            data, "Year", "Year", True
        )

        st.subheader("Environmental Information")

        c1, c2 = st.columns(2)

        with c1:
            rainfall = numeric_input(
                data,
                "average_rain_fall_mm_per_year",
                "Average Annual Rainfall (mm)"
            )

            pesticides = numeric_input(
                data,
                "pesticides_tonnes",
                "Pesticides (tonnes)"
            )

        with c2:
            temperature = numeric_input(
                data,
                "avg_temp",
                "Average Temperature (°C)"
            )

        submitted = st.form_submit_button(
            "Predict Crop Yield",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Area": area,
            "Item": crop,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temperature
        }])

        try:
            prediction = float(model.predict(input_data)[0])
            save_activity("YieldWise", prediction)

            st.markdown(f"""
            <div class="result-card">
                <h2>📈 Predicted Crop Yield</h2>
                <h1>{prediction:,.2f}</h1>
                <p>hg/ha (hectograms per hectare)</p>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "This is a historical-data-based model estimate, "
                "not a guaranteed future yield."
            )

        except Exception as error:
            logger.exception("YieldWise prediction failed")
            st.error(f"Yield prediction failed: {error}")

    show_module_chart("YieldWise")


# ============================================================
# RAINWISE - NEXT-DAY RAINFALL PREDICTION
# ============================================================

def rainwise_page():

    st.title("🌧️ RainWise")
    st.write("Next-day rainfall prediction using machine learning.")

    st.info(
        "Enter the current day's weather details to estimate "
        "rainfall for the following day."
    )

    cities = [
        "Ahmedabad",
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Jaipur",
        "Kolkata",
        "Lucknow",
        "Mumbai",
        "Pune"
    ]

    with st.form("rainwise_form"):

        st.subheader("Location and Date")

        col1, col2 = st.columns(2)

        with col1:
            city = st.selectbox(
                "Select City",
                cities,
                index=cities.index("Chennai")
            )

        with col2:
            observation_date = st.date_input(
                "Observation Date",
                value=pd.Timestamp.today().date()
            )

        st.divider()
        st.subheader("Temperature Details")

        col1, col2 = st.columns(2)

        with col1:
            temp_max = st.number_input(
                "Maximum Temperature (°C)",
                min_value=-90.0,
                max_value=65.0,
                value=35.2,
                step=0.1
            )

            temp_min = st.number_input(
                "Minimum Temperature (°C)",
                min_value=-90.0,
                max_value=65.0,
                value=26.4,
                step=0.1
            )

        with col2:
            apparent_max = st.number_input(
                "Maximum Apparent Temperature (°C)",
                min_value=-90.0,
                max_value=65.0,
                value=39.1,
                step=0.1
            )

            apparent_min = st.number_input(
                "Minimum Apparent Temperature (°C)",
                min_value=-90.0,
                max_value=65.0,
                value=28.2,
                step=0.1
            )

        st.divider()
        st.subheader("Wind Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            wind_speed = st.number_input(
                "Maximum Wind Speed (km/h)",
                min_value=0.0,
                value=18.5,
                step=0.1
            )

        with col2:
            wind_gust = st.number_input(
                "Maximum Wind Gust (km/h)",
                min_value=0.0,
                value=32.0,
                step=0.1
            )

        with col3:
            wind_direction = st.number_input(
                "Wind Direction (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=240.0,
                step=1.0
            )

        st.divider()
        st.subheader("Rainfall Details")

        rain_sum = st.number_input(
            "Current-Day Rainfall (mm)",
            min_value=0.0,
            value=12.5,
            step=0.1
        )

        submitted = st.form_submit_button(
            "Predict Next-Day Rainfall",
            use_container_width=True
        )

    if submitted:

        if temp_max < temp_min:
            st.error(
                "Maximum temperature cannot be below minimum temperature."
            )
            return

        if apparent_max < apparent_min:
            st.error(
                "Maximum apparent temperature cannot be below minimum."
            )
            return

        try:

            # Load the trained RainWise model
            model_path = (
                BASE_DIR / "models/rainwise/rainwise_best_model.pkl"
            )
            model = joblib.load(model_path)

            # Calculate next-day seasonal features
            target_date = (
                pd.Timestamp(observation_date)
                + pd.Timedelta(days=1)
            )

            day_of_year = target_date.dayofyear

            days_in_year = (
                366 if target_date.is_leap_year else 365
            )

            target_dayofyear_sin = np.sin(
                2 * np.pi * day_of_year / days_in_year
            )

            target_dayofyear_cos = np.cos(
                2 * np.pi * day_of_year / days_in_year
            )

            # Prepare input in the exact training feature order
            features = [
                "city",
                "previous_day_temperature_2m_max",
                "previous_day_temperature_2m_min",
                "previous_day_apparent_temperature_max",
                "previous_day_apparent_temperature_min",
                "previous_day_wind_speed_10m_max",
                "previous_day_wind_gusts_10m_max",
                "previous_day_wind_direction_10m_dominant",
                "previous_day_rain_sum",
                "target_dayofyear_sin",
                "target_dayofyear_cos"
            ]

            input_data = pd.DataFrame(
                [[
                    city,
                    temp_max,
                    temp_min,
                    apparent_max,
                    apparent_min,
                    wind_speed,
                    wind_gust,
                    wind_direction,
                    rain_sum,
                    target_dayofyear_sin,
                    target_dayofyear_cos
                ]],
                columns=features
            )

            # Make prediction
            prediction = float(model.predict(input_data)[0])
            prediction = max(prediction, 0.0)
            save_activity("RainWise", prediction)

            # Display result
            st.markdown(
                """
                <div class="result-card">
                    <h2>RainWise Prediction Result</h2>
                    <p>Estimated rainfall for the following day</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Rainfall",
                    f"{prediction:.2f} mm"
                )

            with col2:
                st.metric(
                    "Prediction Date",
                    target_date.strftime("%d %B %Y")
                )

            st.success(
                f"RainWise estimates {prediction:.2f} mm "
                f"of rainfall for {city}."
            )

            st.caption(
                "This is an ML-based estimate, not a guaranteed "
                "weather forecast. Heavy rainfall may be underestimated."
            )

            with st.expander("View Input Details"):
                st.write("City:", city)
                st.write("Observation Date:", observation_date)
                st.write("Maximum Temperature:", temp_max, "°C")
                st.write("Minimum Temperature:", temp_min, "°C")
                st.write("Current-Day Rainfall:", rain_sum, "mm")

        except Exception:

            logger.exception("RainWise prediction failed")

            st.error(
                "RainWise prediction failed. "
                "Please check the model file and input details."
            )

    show_module_chart("RainWise")

# ============================================================
# LEAFGUARD PAGE
# ============================================================

def leafguard_page():

    st.title("🌿 LeafGuard")
    st.write(
        "Upload a plant leaf image to identify its possible "
        "disease using our trained deep learning model."
    )

    st.info(
        "For best results, upload a clear image containing "
        "one plant leaf."
    )

    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] section {
        border: 2px dashed #6C987B;
        border-radius: 16px;
        padding: 28px;
        background: rgba(108, 152, 123, 0.08);
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: #9BC7A6;
        background: rgba(108, 152, 123, 0.14);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:10px 0 4px;">
        <div style="font-size:42px;">☁️</div>
        <h3>Drag and drop your leaf image here</h3>
        <p style="color:#9aaa9e;">
            Or use Browse files below · JPG, JPEG, PNG
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leafguard_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Leaf Image")
                st.image(image, use_container_width=True)

            with col2:

                st.subheader("Disease Analysis")

                if st.button(
                    "Analyze Leaf",
                    key="leafguard_analyze",
                    type="primary"
                ):

                    with st.spinner("Analyzing leaf image..."):

                        model, checkpoint = load_leafguard_model()

                        predictions = predict_leafguard(
                            image,
                            model,
                            checkpoint
                        )

                    best_prediction = predictions[0]

                    predicted_class = best_prediction["class"]
                    save_activity(
    "LeafGuard",
    f"{predicted_class.replace('___', ' - ').replace('_', ' ')} ({best_prediction['confidence'] * 100:.2f}% confidence)"
)

                    display_class = predicted_class.replace(
                        "___", " - "
                    ).replace("_", " ")

                    confidence = best_prediction["confidence"] * 100

                    st.success("Analysis completed!")

                    st.markdown("### Predicted Result")

                    st.subheader(display_class)

                    st.metric(
                        "Model Confidence Score",
                        f"{confidence:.2f}%"
                    )

                    if confidence < 60:
                        st.warning(
                            "The model has relatively low confidence. "
                            "Consider uploading a clearer image."
                        )

                    st.markdown("### Top 3 Predictions")

                    for prediction in predictions:

                        class_name = prediction["class"].replace(
                            "___", " - "
                        ).replace("_", " ")

                        score = prediction["confidence"] * 100

                        st.write(
                            f"**{class_name}** — {score:.2f}%"
                        )

                        st.progress(
                            prediction["confidence"]
                        )

                    st.caption(
                        "These scores represent the model's outputs, "
                        "not guaranteed diagnostic accuracy."
                    )

                    st.warning(
                        "LeafGuard is trained using PlantVillage images. "
                        "Real-world field images may produce different "
                        "results. Confirm suspected diseases with a "
                        "qualified agricultural expert."
                    )

        except Exception:

            logger.exception("LeafGuard prediction failed")

            st.error(
                "Unable to analyze this image. "
                "Please upload a valid JPG, JPEG or PNG image."
            )

    show_module_chart("LeafGuard")


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Home":
    home_page()

elif page == "Dashboard":
    dashboard_page()

elif page == "CropWise":
    cropwise_page()

elif page == "SoilSense":
    soilsense_page()

elif page == "FertilizerWise":
    fertilizerwise_page()

elif page == "YieldWise":
    yieldwise_page()

elif page == "RainWise":
    rainwise_page()

elif page == "LeafGuard":
    leafguard_page()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:#8eaf99;'>"
    "AgriWise | Smart Agriculture Prediction System using ML"
    "</p>",
    unsafe_allow_html=True
)



#python -m streamlit run app.py
