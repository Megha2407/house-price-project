# ============================================================
# HOUSEPRICE AI - HOUSE PRICE PREDICTION APP
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HousePrice AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "house_price_prediction_pipeline.joblib"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #07111f;
        --surface: rgba(255, 255, 255, 0.065);
        --border: rgba(255, 255, 255, 0.13);
        --text: #f8fafc;
        --muted: #a5b4c8;
        --cyan: #38bdf8;
        --purple: #a78bfa;
        --green: #34d399;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(59, 130, 246, 0.20),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(139, 92, 246, 0.20),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #07111f 0%,
                #0b1728 50%,
                #111827 100%
            );
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.8);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    p, label, span {
        font-family: 'Inter', sans-serif;
    }

    .hero {
        padding: 34px;
        border: 1px solid var(--border);
        border-radius: 24px;
        background:
            linear-gradient(
                120deg,
                rgba(56, 189, 248, 0.12),
                rgba(139, 92, 246, 0.12),
                rgba(255, 255, 255, 0.035)
            );
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.20);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #67e8f9;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: clamp(32px, 5vw, 52px);
        line-height: 1.12;
        font-weight: 800;
        margin: 12px 0;
        background: linear-gradient(
            90deg,
            #ffffff,
            #7dd3fc,
            #c4b5fd
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.8;
        max-width: 750px;
    }

    .pill {
        display: inline-block;
        padding: 8px 13px;
        margin: 5px 6px 0 0;
        border-radius: 30px;
        color: #e0f2fe;
        background: rgba(56, 189, 248, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.25);
        font-size: 12px;
        font-weight: 600;
    }

    .section-card {
        padding: 22px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--surface);
        margin: 12px 0 22px 0;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 18px;
    }

    div[data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 24px;
        background: rgba(255, 255, 255, 0.035);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: rgba(255, 255, 255, 0.06);
        color: #f8fafc;
        border-radius: 10px;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 48px;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.4);
        background: linear-gradient(
            90deg,
            #0284c7,
            #4f46e5
        );
        color: white;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        border-color: #7dd3fc;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.20);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid var(--border);
        padding: 22px;
        border-radius: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #7dd3fc !important;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 28px 0 10px 0;
    }

    hr {
        border-color: rgba(255, 255, 255, 0.10);
    }

    @media (max-width: 700px) {
        .hero {
            padding: 22px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD THE SAVED MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the model artifact saved by train_models.py.

    Expected artifact:
    {
        "pipeline": trained sklearn pipeline,
        "model_name": selected model name,
        "feature_columns": training feature order,
        ...
    }
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}\n\n"
            "Run train_models.py to create the model file."
        )

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        if "pipeline" not in artifact:
            raise KeyError(
                "The model artifact does not contain "
                "the required 'pipeline' key."
            )

        pipeline = artifact["pipeline"]

        model_name = artifact.get(
            "model_name",
            "Trained Regression Model",
        )

        feature_columns = artifact.get(
            "feature_columns",
            None,
        )

    else:
        # Also support a directly saved sklearn pipeline.
        pipeline = artifact
        model_name = "Trained Regression Model"
        feature_columns = None

    return pipeline, model_name, feature_columns


# ============================================================
# 5. FEATURE ENGINEERING
# Must match train_models.py
# ============================================================

def prepare_features(data):
    data = data.copy()

    # Ensure numeric values are valid.
    numeric_columns = [
        "BHK",
        "Bathrooms",
        "Super_Area_SqFt",
        "Carpet_Area_SqFt",
        "Floor_Number",
        "Total_Floors",
        "Age_of_Property",
        "Parking",
        "Lift_Available",
        "Gated_Community",
        "Distance_to_Metro_km",
        "Distance_to_City_Center_km",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Area-related features.
    data["Carpet_Area_Ratio"] = (
        data["Carpet_Area_SqFt"]
        / data["Super_Area_SqFt"].replace(0, np.nan)
    )

    data["Non_Carpet_Area_SqFt"] = (
        data["Super_Area_SqFt"]
        - data["Carpet_Area_SqFt"]
    )

    # Floor-related features.
    data["Floor_Position_Ratio"] = (
        data["Floor_Number"]
        / data["Total_Floors"].replace(0, np.nan)
    )

    data["Is_Ground_Floor"] = (
        data["Floor_Number"] == 0
    ).astype(int)

    data["Is_Top_Floor"] = (
        data["Floor_Number"] == data["Total_Floors"]
    ).astype(int)

    # Combined categorical features.
    data["City_LocalityType"] = (
        data["City"].fillna("Unknown").astype(str)
        + "_"
        + data["Locality_Type"].fillna("Unknown").astype(str)
    )

    data["City_PropertyType"] = (
        data["City"].fillna("Unknown").astype(str)
        + "_"
        + data["Property_Type"].fillna("Unknown").astype(str)
    )

    # Match the training script's handling of invalid ratios.
    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return data


# ============================================================
# 6. APP HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ AI-POWERED PROPERTY VALUATION</div>
        <div class="hero-title">🏠 HousePrice AI</div>
        <div class="hero-description">
            Estimate the value of a residential property using
            machine learning and Indian housing data.
            Enter the property details below to get a predicted
            price in seconds.
        </div>
        <br>
        <span class="pill">🤖 Machine Learning</span>
        <span class="pill">📊 Price Prediction</span>
        <span class="pill">🇮🇳 Indian Housing</span>
        <span class="pill">⚡ Instant Estimate</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. LOAD MODEL AND SHOW MODEL STATUS
# ============================================================

try:
    pipeline, model_name, feature_columns = load_model()
    model_loaded = True

except Exception as error:
    model_loaded = False
    pipeline = None
    model_name = None
    feature_columns = None

    st.error(
        "The trained model could not be loaded."
    )

    st.markdown("### Troubleshooting")

    st.write(
        "Check that the model was trained and saved using "
        "the same Python environment and compatible package "
        "versions used by this Streamlit app."
    )

    st.code(
        str(error),
        language="text",
    )

    st.caption(
        f"Expected model location: {MODEL_PATH}"
    )


if model_loaded:
    st.success(
        f"Model loaded successfully: {model_name}"
    )


# ============================================================
# 8. PROPERTY INPUT FORM
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        📝 Enter Property Details
    </div>
    <div class="section-description">
        Fill in the details as accurately as possible.
        All area values are measured in square feet.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.form("house_prediction_form"):

    # --------------------------------------------------------
    # LOCATION AND PROPERTY TYPE
    # --------------------------------------------------------

    st.markdown("### 📍 Location & Property Type")

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.text_input(
            "City",
            value="Chennai",
            help="Enter the city where the property is located.",
        )

    with col2:
        locality_type = st.selectbox(
            "Locality Type",
            options=[
                "Budget",
                "Mid-Range",
                "Premium",
            ],
        )

    with col3:
        property_type = st.selectbox(
            "Property Type",
            options=[
                "Apartment",
                "Independent House",
                "Builder Floor",
                "Villa",
            ],
        )

    st.divider()

    # --------------------------------------------------------
    # PROPERTY SIZE
    # --------------------------------------------------------

    st.markdown("### 📐 Property Size")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
        )

    with col3:
        super_area = st.number_input(
            "Super Area (Sq Ft)",
            min_value=1.0,
            value=1500.0,
            step=50.0,
        )

    with col4:
        carpet_area = st.number_input(
            "Carpet Area (Sq Ft)",
            min_value=1.0,
            value=1200.0,
            step=50.0,
        )

    st.divider()

    # --------------------------------------------------------
    # BUILDING DETAILS
    # --------------------------------------------------------

    st.markdown("### 🏢 Building Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        floor_number = st.number_input(
            "Floor Number",
            min_value=0,
            max_value=200,
            value=1,
            step=1,
        )

    with col2:
        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
        )

    with col3:
        age = st.number_input(
            "Property Age (Years)",
            min_value=0,
            max_value=200,
            value=5,
            step=1,
        )

    with col4:
        furnishing = st.selectbox(
            "Furnishing Status",
            options=[
                "Unfurnished",
                "Semi-Furnished",
                "Furnished",
            ],
        )

    st.divider()

    # --------------------------------------------------------
    # FACILITIES AND ACCESSIBILITY
    # --------------------------------------------------------

    st.markdown("### 🚗 Facilities & Accessibility")

    col1, col2, col3 = st.columns(3)

    with col1:
        parking = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

    with col2:
        lift_label = st.selectbox(
            "Lift Available",
            options=["Yes", "No"],
        )

        lift_available = (
            1 if lift_label == "Yes" else 0
        )

    with col3:
        gated_label = st.selectbox(
            "Gated Community",
            options=["Yes", "No"],
        )

        gated_community = (
            1 if gated_label == "Yes" else 0
        )

    col1, col2 = st.columns(2)

    with col1:
        distance_to_metro = st.number_input(
            "Distance to Metro (km)",
            min_value=0.0,
            max_value=500.0,
            value=2.0,
            step=0.5,
        )

    with col2:
        distance_to_city = st.number_input(
            "Distance to City Center (km)",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            step=0.5,
        )

    st.divider()

    submitted = st.form_submit_button(
        "✨ Predict House Price",
        type="primary",
        use_container_width=True,
        disabled=not model_loaded,
    )


# ============================================================
# 9. PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not city.strip():
        st.error("Please enter a city.")
        st.stop()

    if carpet_area > super_area:
        st.error(
            "Carpet area cannot be greater than super area."
        )
        st.stop()

    if floor_number > total_floors:
        st.error(
            "Floor number cannot exceed total floors."
        )
        st.stop()

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "City": city.strip(),
                "Locality_Type": locality_type,
                "Property_Type": property_type,
                "BHK": int(bhk),
                "Bathrooms": int(bathrooms),
                "Super_Area_SqFt": float(super_area),
                "Carpet_Area_SqFt": float(carpet_area),
                "Floor_Number": int(floor_number),
                "Total_Floors": int(total_floors),
                "Age_of_Property": int(age),
                "Furnishing_Status": furnishing,
                "Parking": int(parking),
                "Lift_Available": int(lift_available),
                "Gated_Community": int(gated_community),
                "Distance_to_Metro_km": float(
                    distance_to_metro
                ),
                "Distance_to_City_Center_km": float(
                    distance_to_city
                ),
            }
        ]
    )

    # --------------------------------------------------------
    # PREPARE FEATURES AND PREDICT
    # --------------------------------------------------------

    try:
        prepared_data = prepare_features(input_data)

        # Match the exact feature order used in training.
        if feature_columns is not None:

            missing_features = [
                column
                for column in feature_columns
                if column not in prepared_data.columns
            ]

            if missing_features:
                st.error(
                    "The app is missing features required "
                    "by the saved model:"
                )

                st.write(missing_features)
                st.stop()

            prepared_data = prepared_data[
                feature_columns
            ]

        with st.spinner(
            "Analyzing property details..."
        ):
            prediction = float(
                pipeline.predict(prepared_data)[0]
            )

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## 🏡 Estimated Property Price")

        if not np.isfinite(prediction):
            st.error(
                "The model returned an invalid prediction. "
                "Please check the model and input data."
            )
            st.stop()

        if prediction < 0:
            st.warning(
                f"The model returned a negative estimate: "
                f"₹{prediction:,.2f} lakhs. "
                "This is not a valid property valuation."
            )
            st.stop()

        crore_value = prediction / 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Estimated Price",
                value=f"₹{prediction:,.2f} Lakhs",
            )

        with col2:
            st.metric(
                label="Estimated Price in Crores",
                value=f"₹{crore_value:,.2f} Crores",
            )

        st.success(
            "Your property price prediction is ready!"
        )

        st.info(
            "This is an ML-generated estimate, not an "
            "official property valuation. Actual market "
            "prices may differ."
        )

        # ----------------------------------------------------
        # SHOW PROPERTY DETAILS
        # ----------------------------------------------------

        with st.expander(
            "📋 View Submitted Property Details"
        ):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "⚙️ View Engineered Model Features"
        ):
            st.dataframe(
                prepared_data,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        with st.expander("🤖 About the Model"):
            st.write(f"**Model:** {model_name}")
            st.write("**Problem type:** Regression")
            st.write("**Target:** House price in INR lakhs")
            st.write(
                f"**Input features:** "
                f"{len(feature_columns) if feature_columns else len(prepared_data.columns)}"
            )

    except Exception as error:
        st.error(
            "Prediction failed. The input features or "
            "saved model may be incompatible."
        )

        st.code(
            str(error),
            language="text",
        )


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <hr>
        🏠 <b>HousePrice AI</b><br>
        Machine Learning Powered Property Valuation<br>
        Built with Python · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)