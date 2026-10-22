from pathlib import Path
from textwrap import dedent
import html
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bengaluru Homes | Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "cleaned_bengaluru_house_data.csv"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def html_block(markup: str) -> None:
    """Render HTML safely from a dedented string (avoids Markdown code blocks)."""
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained prediction model…")
def load_model(path_string: str):
    return joblib.load(path_string)


@st.cache_data(show_spinner="Loading Bengaluru property locations…")
def load_data(path_string: str):
    frame = pd.read_csv(path_string)
    required = {"area_type", "location", "BHK", "total_sqft", "bath", "balcony", "price_in_lakhs"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"The cleaned dataset is missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["location"] = frame["location"].astype(str).str.strip()
    frame = frame[frame["location"].ne("")]
    locations = sorted(frame["location"].dropna().unique().tolist())
    area_types = sorted(frame["area_type"].dropna().astype(str).unique().tolist())
    return frame, locations, area_types


def identify_model(model) -> str:
    """Try to describe the saved estimator without guessing its model family."""
    candidates = [model]
    if hasattr(model, "named_steps"):
        candidates.extend(list(model.named_steps.values()))
    elif hasattr(model, "steps"):
        candidates.extend(step for _, step in model.steps)
    names = [type(item).__name__ for item in candidates]
    for name in reversed(names):
        lower = name.lower()
        if "xgb" in lower:
            return "XGBoost"
        if "randomforest" in lower:
            return "Random Forest"
        if "gradientboost" in lower:
            return "Gradient Boosting"
        if "decisiontree" in lower:
            return "Decision Tree"
        if "linearregression" in lower:
            return "Linear Regression"
        if "ridge" in lower:
            return "Ridge Regression"
        if "lasso" in lower:
            return "Lasso Regression"
        if "catboost" in lower:
            return "CatBoost"
        if "lgbm" in lower or "lightgbm" in lower:
            return "LightGBM"
    return "Trained Machine Learning Model"


# ------------------------------------------------------------
# STYLING — CSS only; layout and form remain native Streamlit
# ------------------------------------------------------------
st.markdown(dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { color-scheme: dark; --bg:#080d1b; --panel:#111a2e; --line:rgba(148,171,220,.18); --text:#f1f5ff; --muted:#a7b4ce; --blue:#65a7ff; --purple:#a78bfa; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { color:var(--text); background:radial-gradient(ellipse at 8% 0%,rgba(58,104,210,.22),transparent 35%),radial-gradient(ellipse at 92% 20%,rgba(139,92,246,.17),transparent 32%),linear-gradient(135deg,#080d1b 0%,#0b1224 55%,#0c1020 100%); }
[data-testid="stHeader"] {
    background:transparent !important;
    box-shadow:none !important;
}
[data-testid="stToolbar"] { background:transparent !important; }
section[data-testid="stMain"] > div.block-container {
    padding-top:0.75rem !important;
}
.block-container { max-width:1220px; padding-top:0.75rem; padding-bottom:3.5rem; }
#MainMenu, footer { visibility:hidden; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--text) !important; }
p, label, [data-testid="stCaptionContainer"] { color:var(--muted); }
.hero { position:relative; overflow:hidden; padding:clamp(28px,5vw,62px); border:1px solid rgba(128,161,255,.28); border-radius:30px; background:linear-gradient(120deg,rgba(33,54,99,.94),rgba(35,31,77,.91)); box-shadow:0 22px 70px rgba(0,0,0,.22); animation:rise .65s ease-out both; }
.hero:after { content:''; position:absolute; width:320px; height:320px; right:-90px; top:-150px; border-radius:50%; background:rgba(101,167,255,.14); filter:blur(3px); pointer-events:none; }
.eyebrow { display:inline-flex; gap:8px; align-items:center; padding:8px 14px; border:1px solid rgba(101,167,255,.34); border-radius:999px; background:rgba(101,167,255,.1); color:#c5ddff; font-size:12px; font-weight:700; letter-spacing:1.1px; }
.hero h1 { max-width:800px; margin:24px 0 14px; font-size:clamp(36px,5.4vw,64px); line-height:1.08; letter-spacing:-2px; }
.gradient { background:linear-gradient(90deg,#8bc5ff,#c1a8ff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-copy { max-width:690px; font-size:17px; line-height:1.8; color:#c4cde2; }
.hero-tag { display:inline-block; margin-top:17px; padding:10px 15px; border:1px solid rgba(255,255,255,.12); border-radius:13px; background:rgba(255,255,255,.07); color:#dce6ff; font-size:14px; }
.feature-card { height:100%; min-height:158px; padding:24px; border:1px solid var(--line); border-radius:23px; background:linear-gradient(145deg,rgba(20,31,53,.96),rgba(14,23,41,.96)); transition:transform .2s ease,border-color .2s ease; }
.feature-card:hover { transform:translateY(-3px); border-color:rgba(101,167,255,.45); }
.feature-icon { font-size:25px; margin-bottom:17px; }
.feature-title { color:#f2f5ff; font-weight:800; font-size:18px; margin-bottom:9px; }
.feature-description { color:#aebbd3; font-size:14px; line-height:1.7; }
.section-heading { margin:38px 0 5px; font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; letter-spacing:-.7px; color:#f1f5ff; }
.section-subtitle { margin-bottom:22px; color:#a7b4ce; font-size:16px; }
.form-panel { padding:24px 25px 8px; border:1px solid var(--line); border-radius:25px; background:rgba(17,26,46,.76); }
[data-testid="stForm"] { padding:22px 24px 20px; border:1px solid var(--line); border-radius:24px; background:rgba(17,26,46,.8); }
[data-testid="stWidgetLabel"] p { color:#e0e7f8 !important; font-weight:650; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background-color:#151f34 !important; border-color:#2b3b5d !important; border-radius:11px !important; }
[data-baseweb="select"] *, [data-baseweb="input"] input { color:#f1f5ff !important; }
.stNumberInput button { color:#dce7ff !important; background:#1a2944 !important; border-color:#2b3b5d !important; }
.stButton > button, [data-testid="stFormSubmitButton"] button {
  width:100%; min-height:54px; border:0 !important; border-radius:12px !important;
  padding:.9rem 1.2rem !important; color:#081225 !important;
  -webkit-text-fill-color:#081225 !important; font-weight:800 !important;
  background:linear-gradient(95deg,#83c1ff,#b6a2ff) !important;
  opacity:1 !important; box-shadow:0 8px 22px rgba(101,167,255,.16);
  transition:transform .2s ease,filter .2s ease;
}
.stButton > button *, [data-testid="stFormSubmitButton"] button * {
  color:#081225 !important; -webkit-text-fill-color:#081225 !important;
  opacity:1 !important; font-weight:800 !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  transform:translateY(-2px); filter:brightness(1.06);
}
.stButton > button:focus, [data-testid="stFormSubmitButton"] button:focus {
  outline:3px solid rgba(131,193,255,.45) !important; outline-offset:3px;
}
.result-card { margin-top:24px; padding:30px; border:1px solid rgba(101,167,255,.35); border-radius:24px; background:radial-gradient(ellipse at 100% 0%,rgba(167,139,250,.16),transparent 40%),linear-gradient(135deg,#142443,#151b38); animation:rise .45s ease-out both; }
.result-label { color:#c4d8ff; font-size:14px; font-weight:700; letter-spacing:.6px; }
.result-price { margin:10px 0 3px; font-family:'Manrope',sans-serif; color:#fff; font-size:clamp(34px,5vw,52px); font-weight:800; letter-spacing:-1.6px; }
.result-crore { color:#b9c8e6; font-size:17px; }
.result-note { margin-top:18px; padding-top:16px; border-top:1px solid rgba(255,255,255,.12); color:#aebbd3; line-height:1.8; }
.footer { margin-top:44px; padding:24px 8px 0; border-top:1px solid var(--line); color:#7787a5; text-align:center; font-size:12px; line-height:1.9; letter-spacing:.25px; }
div[data-testid="stMetric"] { padding:16px; border:1px solid var(--line); border-radius:16px; background:rgba(17,26,46,.72); }
@keyframes rise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@media(max-width:700px) { .block-container{padding:1rem 1rem 2rem;} .hero{border-radius:22px;padding:25px 22px;} .hero h1{letter-spacing:-1px;} [data-testid="stForm"]{padding:16px;} .section-heading{font-size:25px;} }
</style>
"""), unsafe_allow_html=True)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
html_block("""
<div class="hero">
  <div class="eyebrow">✦ SMART REAL ESTATE ESTIMATOR</div>
  <h1>Find your next home<br>in <span class="gradient">Bengaluru.</span></h1>
  <div class="hero-copy">Discover an estimated property price using machine learning. Choose your location, configure your property details, and get a personalized estimate in seconds.</div>
  <div class="hero-tag">🤖 Machine Learning &nbsp; • &nbsp; 📍 Bengaluru, India &nbsp; • &nbsp; ⚡ Instant Estimate</div>
</div>
""")

st.write("")
feature_cols = st.columns(3, gap="medium")
features = [
    ("📍", "Location-aware", "Choose from locations available in the cleaned Bengaluru housing dataset."),
    ("🧠", "ML-powered", "Your saved, trained model generates an estimate from the property details you provide."),
    ("⚡", "Instant estimate", "Submit the form to see an estimated price in lakhs and crores."),
]
for col, (icon, title, description) in zip(feature_cols, features):
    with col:
        html_block(f"""
        <div class="feature-card">
          <div class="feature-icon">{icon}</div>
          <div class="feature-title">{title}</div>
          <div class="feature-description">{description}</div>
        </div>
        """)

# ------------------------------------------------------------
# LOAD FILES WITH FRIENDLY ERROR MESSAGES
# ------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info("Place cleaned_bengaluru_house_data.csv inside the project's data folder.")
    st.stop()
if not MODEL_PATH.exists():
    st.error(f"Trained model file not found: {MODEL_PATH}")
    st.info("Place house_price_model.pkl inside the project's models folder.")
    st.stop()

try:
    data, locations, area_types = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
    model_label = identify_model(model)
except Exception as exc:
    st.error("The model or dataset could not be loaded.")
    st.exception(exc)
    st.stop()

# ------------------------------------------------------------
# PROPERTY FORM
# ------------------------------------------------------------
html_block("""
<div class="section-heading">Configure your property</div>
<div class="section-subtitle">Enter the details below to estimate the property price.</div>
""")

with st.form("property_prediction_form", clear_on_submit=False):
    left, right = st.columns(2, gap="large")
    with left:
        area_type = st.selectbox("Property area type", options=area_types, help="Select the property area category from the training dataset.")
        location = st.selectbox("Location", options=locations, index=None, placeholder="Search or choose a location", help="Only locations present in the cleaned dataset are listed.")
        bhk = st.select_slider("Number of bedrooms (BHK)", options=list(range(1, 11)), value=2, help="Choose the number of bedrooms.")
    with right:
        total_sqft = st.number_input("Total area (square feet)", min_value=300, max_value=20000, value=650, step=50, help="Enter the total built-up area in square feet.")
        bath = st.select_slider("Number of bathrooms", options=list(range(1, 11)), value=1)
        balcony = st.select_slider("Number of balconies", options=list(range(0, 6)), value=1)
    submitted = st.form_submit_button("✨  Estimate property price", use_container_width=True)

if submitted:
    if not location:
        st.warning("Please choose a location before predicting the price.")
    else:
        input_data = pd.DataFrame([{
            "area_type": area_type,
            "location": location,
            "BHK": int(bhk),
            "total_sqft": float(total_sqft),
            "bath": int(bath),
            "balcony": int(balcony),
        }])
        try:
            with st.spinner("Analyzing your property details…"):
                prediction = float(model.predict(input_data)[0])
            if not pd.notna(prediction):
                st.error("The model returned an invalid estimate. Please try again.")
            elif prediction <= 0:
                st.warning("The model returned a non-positive estimate. Try adjusting the property details.")
            else:
                safe_location = html.escape(location)
                html_block(f"""
                <div class="result-card">
                  <div class="result-label">✦ ESTIMATED PROPERTY PRICE</div>
                  <div class="result-price">₹ {prediction:,.2f} Lakhs</div>
                  <div class="result-crore">Approximately ₹ {prediction / 100:,.2f} Crore</div>
                  <div class="result-note">📍 {safe_location} &nbsp; • &nbsp; {int(bhk)} BHK &nbsp; • &nbsp; {int(total_sqft):,} sq. ft.<br>This is a machine-learning estimate, not an official property valuation.</div>
                </div>
                """)
                st.write("")
                st.subheader("Your property summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Bedrooms", f"{int(bhk)} BHK")
                m2.metric("Property area", f"{int(total_sqft):,} sq. ft.")
                m3.metric("Bathrooms", str(int(bath)))
                m4.metric("Balconies", str(int(balcony)))
                if prediction > 430:
                    st.info("This estimate is above ₹430 lakhs. If your training data excluded properties above this price, the estimate may be less reliable at this level.")
                st.caption(f"Model detected from saved estimator: {model_label} · Output: price in lakhs · Estimates depend on training data coverage.")
        except Exception as exc:
            st.error("Prediction failed. Check that the saved model was trained with the same six input features and column names.")
            st.exception(exc)

# ------------------------------------------------------------
# DATASET SNAPSHOT / FOOTER
# ------------------------------------------------------------
st.write("")
with st.expander("About the dataset used by this app"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Cleaned records", f"{len(data):,}")
    c2.metric("Available locations", f"{len(locations):,}")
    c3.metric("Area types", f"{len(area_types):,}")
    st.caption("These counts are read directly from data/cleaned_bengaluru_house_data.csv. Model performance depends on the training process and data quality.")

html_block("""
<div class="footer">
  BENGALURU HOMES · HOUSE PRICE PREDICTION<br>
  Built with Python, Pandas, Joblib and Streamlit<br>
  For educational purposes only · Not financial or official valuation advice
</div>
""")
