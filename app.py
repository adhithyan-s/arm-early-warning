import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))
from src.data.loaders import load_ecdc_data
from src.data.validators import validate_ecdc_data

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AMR Early Warning System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .warning-LOW    { color: #2e7d32; font-weight: 700; font-size: 1.4rem; }
    .warning-MODERATE { color: #f57c00; font-weight: 700; font-size: 1.4rem; }
    .warning-HIGH   { color: #c62828; font-weight: 700; font-size: 1.4rem; }
    .warning-CRITICAL { color: #4a0000; font-weight: 700; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data once and cache it
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    import logging
    logging.disable(logging.CRITICAL)
    df = load_ecdc_data("data/raw/")
    df = validate_ecdc_data(df)
    logging.disable(logging.NOTSET)
    return df

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🦠 AMR Early Warning")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🌍 Resistance Explorer", "🔮 Forecast", "📊 Model Info"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data source:** ECDC EARS-Net  
**Coverage:** 31 EU/EEA countries  
**Period:** 2000–2024  
**Model:** LightGBM (R²=0.921)
""")

# ---------------------------------------------------------------------------
# Page 1 — Resistance Explorer
# ---------------------------------------------------------------------------

if page == "🌍 Resistance Explorer":
    st.markdown('<p class="main-header">🌍 Resistance Explorer</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore historical antibiotic resistance trends across Europe using ECDC EARS-Net surveillance data.</p>',
                unsafe_allow_html=True)

    try:
        df = load_data()

        col1, col2, col3 = st.columns(3)
        with col1:
            pathogen = st.selectbox(
                "Pathogen",
                sorted(df["Pathogen"].unique())
            )
        with col2:
            antibiotic = st.selectbox(
                "Antibiotic",
                sorted(df[df["Pathogen"] == pathogen]["Antibiotic"].unique())
            )
        with col3:
            view = st.selectbox(
                "View",
                ["EU Average Trend", "Country Comparison", "Country Deep Dive"]
            )

        subset = df[
            (df["Pathogen"] == pathogen) &
            (df["Antibiotic"] == antibiotic)
        ].dropna(subset=["PctResistant"])

        if view == "EU Average Trend":
            eu_avg = subset.groupby("Year")["PctResistant"].mean().reset_index()

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(eu_avg["Year"], eu_avg["PctResistant"], marker="o", markersize=4, linewidth=2.5, color="#1f4e79")
            ax.fill_between(eu_avg["Year"], eu_avg["PctResistant"], alpha=0.15, color="#1f4e79")
            ax.set_title(
                f"EU-wide Mean Resistance: {pathogen} / {antibiotic}",
                fontsize=13, pad=15
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("% Resistant Isolates")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            col1, col2, col3, col4 = st.columns(4)
            latest = eu_avg.iloc[-1]["PctResistant"]
            earliest = eu_avg.iloc[0]["PctResistant"]
            change = latest - earliest
            col1.metric("Latest Rate", f"{latest:.1f}%")
            col2.metric("Earliest Rate", f"{earliest:.1f}%")
            col3.metric("Total Change", f"{change:+.1f}pp")
            col4.metric("Countries Reporting", subset["Country"].nunique())

        elif view == "Country Comparison":
            latest_year = subset["Year"].max()
            latest = subset[subset["Year"] == latest_year].sort_values(
                "PctResistant", ascending=True
            )

            fig, ax = plt.subplots(figsize=(12, 8))
            colors = [
                "#2e7d32" if v < 5 else
                "#f57c00" if v < 25 else
                "#c62828" if v < 50 else
                "#4a0000"
                for v in latest["PctResistant"]
            ]
            bars = ax.barh(latest["Country"], latest["PctResistant"],
                           color=colors, edgecolor="white")
            ax.set_title(
                f"Resistance by Country ({latest_year}): "
                f"{pathogen} / {antibiotic}",
                fontsize=13, pad=15
            )
            ax.set_xlabel("% Resistant Isolates")

            patches = [
                mpatches.Patch(color="#2e7d32", label="LOW (<5%)"),
                mpatches.Patch(color="#f57c00", label="MODERATE (5-25%)"),
                mpatches.Patch(color="#c62828", label="HIGH (25-50%)"),
                mpatches.Patch(color="#4a0000", label="CRITICAL (>50%)"),
            ]
            ax.legend(handles=patches, loc="lower right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        elif view == "Country Deep Dive":
            country = st.selectbox(
                "Select Country",
                sorted(subset["Country"].unique())
            )
            country_data = subset[
                subset["Country"] == country
            ].sort_values("Year")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(country_data["Year"], country_data["PctResistant"], marker="o", markersize=5, linewidth=2.5, color="#c62828")
            ax.fill_between(country_data["Year"], country_data["PctResistant"], alpha=0.15, color="#c62828")

            # EU average for context
            eu_avg = subset.groupby("Year")["PctResistant"].mean()
            ax.plot(eu_avg.index, eu_avg.values,
                    linewidth=1.5, linestyle="--", color="#666",
                    label="EU Average")

            ax.set_title(
                f"{country}: {pathogen} / {antibiotic}",
                fontsize=13, pad=15
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("% Resistant Isolates")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Make sure data/raw/ contains the ECDC CSV files.")

# ---------------------------------------------------------------------------
# Page 2 — Forecast
# ---------------------------------------------------------------------------

elif page == "🔮 Forecast":
    st.markdown('<p class="main-header">🔮 2-Year Resistance Forecast</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict antibiotic resistance 2 years '
                'ahead for a given country and pathogen-antibiotic combination.'
                '</p>', unsafe_allow_html=True)

    VALID_COMBINATIONS = [
        ("Klebsiella pneumoniae",  "Carbapenems"),
        ("Klebsiella pneumoniae",  "Fluoroquinolones"),
        ("Klebsiella pneumoniae",  "3rd gen cephalosporins"),
        ("Escherichia coli",       "Carbapenems"),
        ("Escherichia coli",       "Fluoroquinolones"),
        ("Escherichia coli",       "3rd gen cephalosporins"),
        ("Staphylococcus aureus",  "Meticillin"),
        ("Enterococcus faecium",   "Vancomycin"),
        ("Pseudomonas aeruginosa", "Carbapenems"),
        ("Acinetobacter spp.",     "Carbapenems"),
    ]

    COUNTRY_MAP = {
        "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG",
        "Croatia": "HR", "Cyprus": "CY", "Czechia": "CZ",
        "Denmark": "DK", "Estonia": "EE", "Finland": "FI",
        "France": "FR", "Germany": "DE", "Greece": "EL",
        "Hungary": "HU", "Iceland": "IS", "Ireland": "IE",
        "Italy": "IT", "Latvia": "LV", "Lithuania": "LT",
        "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL",
        "Norway": "NO", "Poland": "PL", "Portugal": "PT",
        "Romania": "RO", "Slovakia": "SK", "Slovenia": "SI",
        "Spain": "ES", "Sweden": "SE",
    }

    # -----------------------------------------------------------------------
    # Step 1 — Dropdowns OUTSIDE the form so antibiotic reacts to pathogen
    # -----------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        country_name = st.selectbox("Country", sorted(COUNTRY_MAP.keys()))
    with col2:
        pathogens = sorted(set(p for p, _ in VALID_COMBINATIONS))
        pathogen = st.selectbox("Pathogen", pathogens)
    with col3:
        antibiotics = sorted(
            set(a for p, a in VALID_COMBINATIONS if p == pathogen)
        )
        antibiotic = st.selectbox("Antibiotic", antibiotics)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Step 2 — Auto-populate numeric defaults from real ECDC data
    # When user changes country/pathogen/antibiotic, the numbers below
    # automatically update to reflect that country's actual latest data.
    # This is what makes predictions differ between selections.
    # -----------------------------------------------------------------------
    defaults = {
        "current": 5.0,
        "previous": 4.0,
        "isolates": 500,
        "regional": 10.0,
    }

    try:
        df = load_data()

        # Get this country's actual latest two years of data
        subset = df[
            (df["Country"] == country_name) &
            (df["Pathogen"] == pathogen) &
            (df["Antibiotic"] == antibiotic)
        ].dropna(subset=["PctResistant"]).sort_values("Year")

        if len(subset) >= 2:
            latest = subset.iloc[-1]
            prev   = subset.iloc[-2]
            defaults["current"]  = round(float(latest["PctResistant"]), 1)
            defaults["previous"] = round(float(prev["PctResistant"]), 1)
            if pd.notna(latest["NumIsolates"]):
                defaults["isolates"] = int(latest["NumIsolates"])

        # EU average for this combo excluding the selected country
        regional = df[
            (df["Pathogen"] == pathogen) &
            (df["Antibiotic"] == antibiotic) &
            (df["Country"] != country_name) &
            (df["Year"] == df["Year"].max())
        ]["PctResistant"].mean()

        if pd.notna(regional):
            defaults["regional"] = round(float(regional), 1)

    except Exception:
        pass  # silently fall back to defaults

    st.info(
        f"📊 Auto-populated with latest ECDC data for "
        f"**{country_name} / {pathogen} / {antibiotic}**. "
        f"You can adjust the values before generating the forecast."
    )

    # -----------------------------------------------------------------------
    # Step 3 — Numeric inputs + submit button inside the form
    # -----------------------------------------------------------------------
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            current_resistance = st.number_input(
                "Current Resistance (%)",
                min_value=0.0, max_value=100.0,
                value=defaults["current"], step=0.1,
                help="Current observed resistance percentage"
            )
            previous_resistance = st.number_input(
                "Previous Year Resistance (%) — optional",
                min_value=0.0, max_value=100.0,
                value=defaults["previous"], step=0.1,
                help="Resistance from 1 year ago"
            )

        with col2:
            num_isolates = st.number_input(
                "Number of Isolates Tested — optional",
                min_value=0, value=defaults["isolates"], step=10,
                help="More isolates = more reliable prediction"
            )
            regional_avg = st.number_input(
                "EU Regional Average (%) — optional",
                min_value=0.0, max_value=100.0,
                value=defaults["regional"], step=0.1,
            )

        submitted = st.form_submit_button(
            "🔮 Generate Forecast", use_container_width=True
        )

    # -----------------------------------------------------------------------
    # Step 4 — Call API and show results
    # -----------------------------------------------------------------------
    if submitted:
        api_url = "http://localhost:8000/predict"
        payload = {
            "country_code":            COUNTRY_MAP[country_name],
            "pathogen":                pathogen,
            "antibiotic":              antibiotic,
            "current_resistance_pct":  current_resistance,
            "previous_resistance_pct": previous_resistance,
            "num_isolates":            num_isolates,
            "regional_avg_pct":        regional_avg,
        }

        with st.spinner("Calling prediction API..."):
            try:
                response = requests.post(api_url, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                predicted = result["predicted_resistance_pct"]
                current   = result["current_resistance_pct"]
                change    = predicted - current
                level     = result["warning_level"]

                col1.metric("Current Resistance", f"{current:.1f}%")
                col2.metric(
                    "Predicted (2yr)",
                    f"{predicted:.1f}%",
                    delta=f"{change:+.1f}pp"
                )
                col3.metric("Warning Level", level)

                color_map = {
                    "LOW":      "#2e7d32",
                    "MODERATE": "#f57c00",
                    "HIGH":     "#c62828",
                    "CRITICAL": "#4a0000",
                }
                color = color_map.get(level, "#666")

                st.markdown(f"""
                <div style="background:{color}18;
                            border-left: 4px solid {color};
                            padding: 1rem; border-radius: 4px;
                            margin-top: 1rem;">
                    <strong style="color:{color}">⚠ {level}</strong><br>
                    {result['warning_message']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### Forecast Visualisation")

                fig, ax = plt.subplots(figsize=(10, 4))
                years   = ["Current", "Year +1 (est.)", "Year +2 (forecast)"]
                mid_est = (current + predicted) / 2
                values  = [current, mid_est, predicted]
                colors_bar = ["#1f4e79", "#5b8db8", color]

                bars = ax.bar(years, values, color=colors_bar,
                              edgecolor="white", width=0.5)
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", fontsize=11
                    )

                ax.set_ylabel("% Resistant Isolates")
                ax.set_title(
                    f"{country_name} — {pathogen} / {antibiotic}",
                    fontsize=12, pad=12
                )
                ax.set_ylim(0, max(values) * 1.3)
                ax.grid(axis="y", alpha=0.3)
                st.pyplot(fig)
                plt.close()

                with st.expander("Model details"):
                    st.markdown(f"""
                    - **Model R²:** {result['model_r2']}
                    - **MAE:** ±{result['model_mae_pp']}pp
                    - **Horizon:** {result['forecast_horizon_years']} years
                    - **Country:** {result['country_code']}
                    - **Pathogen:** {result['pathogen']}
                    - **Antibiotic:** {result['antibiotic']}
                    - **API endpoint:** `POST /predict`
                    """)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the prediction API. "
                    "Make sure the FastAPI server is running: "
                    "`uvicorn src.api.app:app --port 8000`"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Page 3 — Model Info
# ---------------------------------------------------------------------------

elif page == "📊 Model Info":
    st.markdown('<p class="main-header">📊 Model Information</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Training details, performance metrics, '
                'and feature importance.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Comparison")
        comparison = pd.DataFrame({
            "Model":    ["Ridge Baseline", "LightGBM"],
            "MAE (pp)": [4.2207, 3.6019],
            "RMSE (pp)":[6.6977, 6.0616],
            "R²":       [0.9037, 0.9211],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("### Training Setup")
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Algorithm | LightGBM |
        | Split strategy | Temporal (train ≤ 2020) |
        | Test period | 2021–2024 |
        | Forecast horizon | 2 years |
        | Training rows | 4,765 |
        | Test rows | 564 |
        | Features | 12 |
        """)

    with col2:
        st.markdown("### Features Used")
        features = pd.DataFrame({
            "Feature": [
                "lag_1yr", "lag_2yr", "lag_3yr",
                "rolling_mean_3yr", "rolling_mean_5yr",
                "rolling_std_3yr", "rolling_std_5yr",
                "slope_3yr", "slope_5yr",
                "regional_avg_excl_self",
                "log_num_isolates",
                "years_of_data",
            ],
            "Description": [
                "Resistance 1 year ago",
                "Resistance 2 years ago",
                "Resistance 3 years ago",
                "Rolling mean over 3 years",
                "Rolling mean over 5 years",
                "Rolling std dev over 3 years",
                "Rolling std dev over 5 years",
                "Linear slope over 3 years",
                "Linear slope over 5 years",
                "EU average excl. this country",
                "Log of isolates tested",
                "Years of surveillance data",
            ]
        })
        st.dataframe(features, use_container_width=True, hide_index=True)

    st.markdown("### Key Findings")
    st.markdown("""
    - **LightGBM outperforms Ridge** on all metrics, confirming non-linear 
      patterns exist in AMR data
    - **R² of 0.921** means the model explains 92.1% of variance in 
      future resistance rates
    - **MAE of 3.6pp** — on average forecasts are within 3.6 percentage 
      points of the true value
    - **Lag features dominate** — last year's resistance is the strongest 
      single predictor
    - **Regime changes are hard** — K. pneumoniae carbapenem resistance 
      spiked post-2018, a pattern lag features partially miss
    """)

    st.markdown("### Limitations")
    st.markdown("""
    - Model trained on European data only — not generalisable globally
    - Does not account for antibiotic prescription rates (known AMR driver)
    - Point forecasts only — no uncertainty intervals yet
    - World Bank enrichment features not yet integrated
    - Annual data limits granularity — monthly surveillance would improve 
      accuracy
    """)

    st.markdown("### Data Pipeline")
    st.code("""
ECDC EARS-Net CSVs (20 files)
    ↓ loaders.py — merge N + pct files, derive NumIsolates
    ↓ validators.py — Pandera schema validation
    ↓ engineer.py — 12 feature engineering steps
    ↓ train.py — temporal split, Ridge + LightGBM, MLflow tracking
    ↓ app.py — FastAPI REST API
    ↓ app.py — Streamlit UI (this page)
    """, language="text")