import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
 
# ── Custom transformers required to unpickle the model ───────────────────────
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]
 
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]
 
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
 
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
 
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
 
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
 
# Inject into __main__ so pickle can resolve them
_m = sys.modules["__main__"]
_m.column_ratio = column_ratio
_m.ratio_name = ratio_name
_m.ClusterSimilarity = ClusterSimilarity
 
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA House Price Predictor",
    page_icon="🏡",
    layout="wide",
)
 
# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #56ccf2, #2f80ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: rgba(255,255,255,0.55);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        color: #56ccf2;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.4rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(86,204,242,0.15), rgba(47,128,237,0.15));
        border: 1px solid rgba(86,204,242,0.4);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .result-price {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #56ccf2, #2f80ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .insight-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 0.6rem;
    }
    .insight-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.75);
        font-size: 0.88rem;
    }
    .insight-row:last-child { border-bottom: none; }
    .insight-val {
        font-weight: 600;
        color: #56ccf2;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    label { color: rgba(255,255,255,0.8) !important; }
    .stSlider > div > div > div { background: rgba(86,204,242,0.3) !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load("my_california_housing_model.pkl")
 
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()
 
 
# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏡 California House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RandomForest · scikit-learn Pipeline · California Housing Dataset</p>', unsafe_allow_html=True)
 
col_left, col_right = st.columns([1.1, 0.9], gap="large")
 
with col_left:
    # ── Location ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        longitude = st.number_input(
            "Longitude", min_value=-124.35, max_value=-114.31,
            value=-118.25, step=0.01, format="%.4f",
            help="Western CA is around -122 (SF) to -117 (San Diego)"
        )
    with c2:
        latitude = st.number_input(
            "Latitude", min_value=32.54, max_value=41.95,
            value=34.05, step=0.01, format="%.4f",
            help="SF ≈ 37.8, LA ≈ 34.0"
        )
 
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        index=0,
        help="Distance/relationship to the ocean"
    )
 
    # ── Housing ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🏠 Housing Characteristics</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        housing_median_age = st.slider(
            "Median Housing Age (years)", min_value=1, max_value=52,
            value=29, step=1
        )
    with c4:
        median_income = st.slider(
            "Median Income ($10K units)", min_value=0.5, max_value=15.0,
            value=3.87, step=0.1, format="%.2f",
            help="e.g. 5.0 = $50,000 household income"
        )
 
    # ── Population ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">👥 Block-Level Demographics</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        total_rooms = st.number_input("Total Rooms", min_value=1, max_value=40000, value=2635, step=10)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=7000, value=537, step=5)
    with c6:
        population = st.number_input("Population", min_value=1, max_value=40000, value=1425, step=10)
        households = st.number_input("Households", min_value=1, max_value=7000, value=499, step=5)
 
    predict_btn = st.button("🔮 Predict House Price", use_container_width=True, type="primary")
 
 
# ── Prediction ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<p class="section-header">📊 Prediction Result</p>', unsafe_allow_html=True)
 
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])
 
    try:
        predicted_price = model.predict(input_df)[0]
 
        # Derived insights
        rooms_per_house = total_rooms / max(households, 1)
        bedrooms_per_room = total_bedrooms / max(total_rooms, 1)
        people_per_house = population / max(households, 1)
        price_per_room = predicted_price / max(total_rooms, 1)
 
        if predict_btn or True:  # always show live prediction
            st.markdown(f"""
            <div class="result-card">
                <p class="result-label">Estimated Median Home Value</p>
                <p class="result-price">${predicted_price:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
 
            # Derived insights
            st.markdown('<p class="section-header">🔍 Derived Insights</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-row">
                    <span>Rooms per household</span>
                    <span class="insight-val">{rooms_per_house:.1f}</span>
                </div>
                <div class="insight-row">
                    <span>Bedrooms / Total rooms</span>
                    <span class="insight-val">{bedrooms_per_room:.1%}</span>
                </div>
                <div class="insight-row">
                    <span>People per household</span>
                    <span class="insight-val">{people_per_house:.1f}</span>
                </div>
                <div class="insight-row">
                    <span>Price per room (est.)</span>
                    <span class="insight-val">${price_per_room:,.0f}</span>
                </div>
                <div class="insight-row">
                    <span>Median income</span>
                    <span class="insight-val">${median_income * 10_000:,.0f}/yr</span>
                </div>
                <div class="insight-row">
                    <span>Price-to-income ratio</span>
                    <span class="insight-val">{predicted_price / (median_income * 10_000):.1f}×</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
            # Context bar
            st.markdown('<p class="section-header">📈 CA Price Context</p>', unsafe_allow_html=True)
            ca_median = 233_000  # Dataset median (~1990s)
            pct = min(predicted_price / 500_000, 1.0)
            st.progress(pct, text=f"${predicted_price:,.0f} vs CA dataset max ~$500K")
            if predicted_price < ca_median:
                st.info(f"🟢 **Below** dataset median (${ca_median:,})")
            elif predicted_price < ca_median * 1.5:
                st.warning(f"🟡 **Near** dataset median (${ca_median:,})")
            else:
                st.error(f"🔴 **Well above** dataset median (${ca_median:,})")
 
    except Exception as e:
        st.error(f"Prediction failed: {e}")
 
# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: RandomForestRegressor · "
    "Data: California Housing (1990 Census) · "
    "Pipeline includes ClusterSimilarity, ratio features, log transforms, StandardScaler, OHE"
)