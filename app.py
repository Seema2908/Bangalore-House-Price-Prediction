
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Bengaluru Homes | Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "cleaned_bengaluru_house_data.csv"


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_locations():
    data = pd.read_csv(DATA_PATH)

    locations = sorted(
        data["location"].dropna().astype(str).unique().tolist()
    )
    area_types = sorted(
        data["area_type"].dropna().astype(str).unique().tolist()
    )

    return locations, area_types


# --------------------------------------------------
# CUSTOM CSS: DARK THEME + ANIMATIONS
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: dark;
        --background: #080d1b;
        --surface: #111a2e;
        --text: #f1f5ff;
        --muted: #a7b4ce;
        --blue: #65a7ff;
        --purple: #a78bfa;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(58, 104, 210, 0.22), transparent 35%),
            radial-gradient(ellipse at 90% 25%, rgba(139, 92, 246, 0.17), transparent 32%),
            linear-gradient(135deg, #080d1b 0%, #0b1224 50%, #0c1020 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 27, 0.75);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Global text visibility */
    h1, h2, h3, h4, p, label, li, span, div {
        font-family: 'DM Sans', sans-serif;
    }

    .stMarkdown, .stMarkdown p {
        color: #e8eeff;
    }

    [data-testid="stCaptionContainer"] {
        color: #a7b4ce !important;
    }

    /* Animated top label */
    .eyebrow {
        display: inline-block;
        padding: 8px 15px;
        margin-bottom: 20px;
        color: #b9d7ff;
        background: rgba(72, 133, 255, 0.12);
        border: 1px solid rgba(101, 167, 255, 0.35);
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.6px;
        animation: fadeUp 0.8s ease-out both;
    }

    /* Hero section */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 42px 38px;
        margin: 10px 0 28px;
        border: 1px solid rgba(157, 180, 255, 0.18);
        border-radius: 28px;
        background:
            linear-gradient(115deg, rgba(28, 48, 91, 0.96), rgba(27, 24, 62, 0.93));
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.24);
        animation: fadeUp 0.8s ease-out both;
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        top: -150px;
        right: -40px;
        border-radius: 50%;
        background: rgba(86, 150, 255, 0.24);
        filter: blur(65px);
        animation: floatGlow 7s ease-in-out infinite alternate;
        pointer-events: none;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 200px;
        height: 200px;
        right: 28%;
        bottom: -140px;
        border-radius: 50%;
        background: rgba(167, 139, 250, 0.22);
        filter: blur(50px);
        animation: floatGlow 8s ease-in-out infinite alternate-reverse;
        pointer-events: none;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(32px, 5vw, 52px);
        font-weight: 800;
        line-height: 1.16;
        letter-spacing: -1.8px;
        color: #ffffff !important;
    }

    .hero h1 .gradient {
        background: linear-gradient(90deg, #83c5ff, #c4b5fd);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 650px;
        margin-top: 16px;
        color: #d1ddf5 !important;
        font-size: 16px;
        line-height: 1.8;
    }

    .hero-tag {
        position: relative;
        z-index: 1;
        display: inline-flex;
        margin-top: 14px;
        padding: 9px 13px;
        border-radius: 12px;
        background: rgba(255,255,255,0.07);
        color: #e3eaff;
        font-size: 12px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    /* Feature cards */
    .feature-card {
        height: 100%;
        min-height: 110px;
        padding: 20px;
        border-radius: 19px;
        background: linear-gradient(
            145deg,
            rgba(21, 32, 56, 0.96),
            rgba(15, 23, 42, 0.92)
        );
        border: 1px solid rgba(143, 169, 220, 0.16);
        transition: transform 0.28s ease, border-color 0.28s ease,
                    box-shadow 0.28s ease;
        animation: fadeUp 0.8s ease-out both;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(101, 167, 255, 0.55);
        box-shadow: 0 15px 35px rgba(34, 75, 150, 0.15);
    }

    .feature-icon {
        font-size: 23px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #ffffff !important;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .feature-description {
        color: #aebbd5 !important;
        font-size: 12px;
        line-height: 1.6;
    }

    /* Section headings */
    .section-heading {
        color: #f5f7ff !important;
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin: 28px 0 6px;
    }

    .section-subtitle {
        color: #aab8d3 !important;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Form container */
    [data-testid="stForm"] {
        padding: 28px;
        border: 1px solid rgba(143, 169, 220, 0.20);
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.86);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.16);
        animation: fadeUp 0.9s ease-out both;
    }

    /* Labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stSelectbox label,
    .stNumberInput label {
        color: #eaf0ff !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    /* Input controls */
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        color: #f4f7ff !important;
        background-color: #101a30 !important;
        border: 1px solid #354462 !important;
        border-radius: 12px !important;
        min-height: 46px;
    }

    [data-baseweb="select"] > div:hover,
    [data-testid="stNumberInput"] input:hover {
        border-color: #76adff !important;
    }

    [data-baseweb="select"] input {
        color: #ffffff !important;
    }

    [data-baseweb="select"] svg {
        fill: #c5d4f1 !important;
    }

    [data-testid="stNumberInput"] button {
        color: #eaf0ff !important;
        background: #1b2944 !important;
        border-color: #354462 !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #111b30 !important;
        border: 1px solid #344767 !important;
    }

    [role="option"] {
        color: #f1f5ff !important;
        background: #111b30 !important;
    }

    [role="option"]:hover,
    [aria-selected="true"] {
        background: #263b60 !important;
        color: #ffffff !important;
    }

    /* Primary button */
    .stFormSubmitButton button,
    .stButton button {
        width: 100%;
        min-height: 49px;
        border: 1px solid rgba(170, 204, 255, 0.35) !important;
        border-radius: 13px !important;
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        background: linear-gradient(100deg, #3478f6, #7658e8) !important;
        box-shadow: 0 8px 24px rgba(72, 103, 240, 0.24);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }

    .stFormSubmitButton button:hover,
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(86, 113, 255, 0.38);
        border-color: #b3cfff !important;
    }

    .stFormSubmitButton button:active,
    .stButton button:active {
        transform: scale(0.98);
    }

    /* Prediction result */
    .result-card {
        position: relative;
        overflow: hidden;
        margin-top: 28px;
        padding: 32px;
        border: 1px solid rgba(111, 190, 255, 0.40);
        border-radius: 24px;
        background:
            radial-gradient(circle at 100% 0%, rgba(91, 125, 255, 0.22), transparent 35%),
            linear-gradient(135deg, #142847, #171a3a);
        box-shadow: 0 20px 60px rgba(20, 65, 140, 0.22);
        animation: resultReveal 0.7s cubic-bezier(.2,.8,.2,1) both;
    }

    .result-label {
        color: #b7c9e8 !important;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .result-price {
        margin: 13px 0 7px;
        color: #ffffff !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(34px, 5vw, 51px);
        font-weight: 800;
        letter-spacing: -1.6px;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .result-crore {
        color: #9dc8ff !important;
        font-size: 17px;
        font-weight: 600;
    }

    .result-note {
        margin-top: 18px;
        color: #bdcbe2 !important;
        font-size: 13px;
        line-height: 1.7;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        padding: 17px;
        border: 1px solid rgba(143, 169, 220, 0.17);
        border-radius: 16px;
        background: rgba(14, 23, 41, 0.82);
    }

    [data-testid="stMetricLabel"] {
        color: #b6c5df !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Notices */
    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    /* Footer */
    .footer {
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid rgba(143, 169, 220, 0.15);
        text-align: center;
        color: #8797b6 !important;
        font-size: 12px;
        line-height: 1.8;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes floatGlow {
        from { transform: translate(0, 0) scale(1); }
        to { transform: translate(-25px, 22px) scale(1.18); }
    }

    @keyframes resultReveal {
        from {
            opacity: 0;
            transform: translateY(22px) scale(0.985);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    /* Respect reduced-motion accessibility preferences */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 3rem;
        }

        .hero {
            padding: 28px 22px;
            border-radius: 21px;
        }

        [data-testid="stForm"] {
            padding: 18px;
        }

        .result-card {
            padding: 24px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ SMART REAL ESTATE ESTIMATOR</div>
        <h1>Find your next home<br>in <span class="gradient">Bengaluru.</span></h1>
        <p>
            Discover an estimated property price using machine learning.
            Choose your location, configure your property details, and
            get a personalized price estimate in seconds.
        </p>
        <div class="hero-tag">🤖 Machine Learning &nbsp; • &nbsp;
        📍 Bengaluru, India &nbsp; • &nbsp; ⚡ Instant Estimate</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# FEATURE CARDS
# --------------------------------------------------
feature_columns = st.columns(3)

features = [
    (
        "📍",
        "Location-aware",
        "Select from locations available in the cleaned dataset.",
    ),
    (
        "🧠",
        "ML-powered",
        "Predictions generated by your trained Linear Regression pipeline.",
    ),
    (
        "⚡",
        "Instant estimate",
        "Get a price estimate immediately after submitting the form.",
    ),
]

for column, (icon, title, description) in zip(feature_columns, features):
    with column:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please check that "
        "models/house_price_model.pkl exists."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "Cleaned dataset not found. Please check that "
        "data/cleaned_bengaluru_house_data.csv exists."
    )
    st.stop()

try:
    model = load_model()
    locations, area_types = load_locations()
except Exception as error:
    st.error(f"Unable to load the model or dataset: {error}")
    st.stop()

if not locations or not area_types:
    st.error("No locations or area types were found in the cleaned dataset.")
    st.stop()


# --------------------------------------------------
# PREDICTION FORM
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">Configure your property</div>'
    '<div class="section-subtitle">'
    'Enter the details below to estimate the property price.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("property_prediction_form"):
    first_col, second_col = st.columns(2, gap="large")

    with first_col:
        area_type = st.selectbox(
            "Property area type",
            options=area_types,
            help="Choose the area type used in the training dataset.",
        )

        location = st.selectbox(
            "Location",
            options=locations,
            index=None,
            placeholder="Search or choose a location",
            help="Choose a location from the cleaned dataset.",
        )

        bhk = st.selectbox(
            "Number of bedrooms (BHK)",
            options=list(range(1, 11)),
            index=1,
        )

    with second_col:
        total_sqft = st.number_input(
            "Total area (square feet)",
            min_value=100,
            max_value=30000,
            value=1200,
            step=50,
        )

        bath = st.selectbox(
            "Number of bathrooms",
            options=list(range(1, 13)),
            index=1,
        )

        balcony = st.selectbox(
            "Number of balconies",
            options=list(range(0, 5)),
            index=1,
        )

    submitted = st.form_submit_button(
        "✨  Predict Property Price",
        use_container_width=True,
    )


# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------
if submitted:
    if location is None:
        st.warning("Please choose a location before predicting the price.")
    else:
        input_data = pd.DataFrame(
            [
                {
                    "area_type": area_type,
                    "location": location,
                    "BHK": bhk,
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "balcony": balcony,
                }
            ]
        )

        try:
            with st.spinner("Analyzing your property details..."):
                prediction = float(model.predict(input_data)[0])

            if not pd.notna(prediction):
                st.error("The model returned an invalid prediction.")
            elif prediction <= 0:
                st.warning(
                    "The model returned a non-positive price estimate. "
                    "Try different property details."
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            ✦ Estimated property price
                        </div>
                        <div class="result-price">
                            ₹ {prediction:,.2f} Lakhs
                        </div>
                        <div class="result-crore">
                            Approximately ₹ {prediction / 100:,.2f} Crore
                        </div>
                        <div class="result-note">
                            📍 {location} &nbsp; • &nbsp; {bhk} BHK
                            &nbsp; • &nbsp; {total_sqft:,} sq. ft.<br>
                            This is a machine-learning estimate, not an
                            official property valuation.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-heading">Your property summary</div>',
                    unsafe_allow_html=True,
                )

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric("Bedrooms", f"{bhk} BHK")
                metric2.metric("Property area", f"{total_sqft:,} sq. ft.")
                metric3.metric("Bathrooms", str(bath))

                if prediction > 430:
                    st.info(
                        "This estimate is above ₹430 lakhs. Your model was "
                        "trained after filtering out properties above this "
                        "price threshold, so this prediction may be less "
                        "reliable."
                    )

                st.caption(
                    "Model: Linear Regression | "
                    "Output: price in lakhs | "
                    "Results depend on the quality and coverage of the dataset."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        BENGALURU HOMES · HOUSE PRICE PREDICTION<br>
        Built with Python, Pandas, Scikit-learn and Streamlit<br>
        For educational purposes only · Not financial or valuation advice
    </div>
    """,
    unsafe_allow_html=True,
)
