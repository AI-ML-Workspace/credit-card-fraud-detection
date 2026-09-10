"""
streamlit_app.py
-----------------
FraudShield AI — interactive Streamlit application.

Brings together the src/ modules into a demo-able UI:
  - Single Transaction  : score one transaction, show risk tier + SHAP explanation
  - Batch Upload        : score a CSV of transactions, download results
  - Live Simulator      : stream test-set transactions one at a time (src/simulator.py)
  - Model Insights      : global SHAP summary plot (src/explain.py)

Run with:
    streamlit run app/streamlit_app.py

Requires a trained model + fitted scaler at the paths configured in
src/config.py (models/final_model.pkl, models/scaler.pkl), produced by
running the training pipeline first (see src/train_model.py or the
notebooks/ pipeline).
"""

import sys
from pathlib import Path

# Make sure `src` is importable when Streamlit runs this file directly
# (streamlit run executes the file in isolation, not as part of a package).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src import config, predict, simulator

# ---------------------------------------------------------------------
# Risk tiers
#
# Documented, fixed thresholds on the model's predicted fraud
# probability — NOT learned by the model. HIGH reuses the existing
# config.ALERT_PROBABILITY_THRESHOLD (0.7) so the "alert" flag used by
# the live simulator stays consistent with the risk tier shown here.
# Adjust these two cut points deliberately and re-document them if
# you change them; don't present them as model-derived.
# ---------------------------------------------------------------------
RISK_LOW_MAX = 0.3
RISK_HIGH_MIN = config.ALERT_PROBABILITY_THRESHOLD  # 0.7


def risk_tier(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "Low"
    elif probability < RISK_HIGH_MIN:
        return "Medium"
    return "High"


TIER_COLOR = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


@st.cache_resource(show_spinner=False)
def get_model_and_scaler():
    """Load once per session; surfaces a clear error if artifacts are missing."""
    try:
        model = predict._get_model()
        scaler = predict._get_scaler()
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, str(e)


def require_artifacts():
    """Show a friendly message and stop the page if model/scaler aren't trained yet."""
    _, _, error = get_model_and_scaler()
    if error:
        st.error(
            "No trained model found at "
            f"`{config.FINAL_MODEL_PATH}` (or scaler at `{config.SCALER_PATH}`).\n\n"
            "Run the training pipeline first, e.g.:\n"
            "```bash\npython -m src.preprocess\npython -m src.train_model\n```"
        )
        st.stop()


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="FraudShield AI", page_icon="💳", layout="wide")
st.title("💳 FraudShield AI")
st.caption("Predictive analytics for credit-card fraud detection")

page = st.sidebar.radio(
    "Navigate",
    ["Single Transaction", "Batch Upload", "Live Simulator", "Model Insights"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Risk tiers**\n\n"
    f"🟢 Low: probability < {RISK_LOW_MAX}\n\n"
    f"🟡 Medium: {RISK_LOW_MAX} – {RISK_HIGH_MIN}\n\n"
    f"🔴 High: probability ≥ {RISK_HIGH_MIN}\n\n"
    f"*(fixed thresholds, not model-learned)*"
)

# ---------------------------------------------------------------------
# Page: Single Transaction
# ---------------------------------------------------------------------
if page == "Single Transaction":
    require_artifacts()
    st.subheader("Score a single transaction")

    with st.form("single_transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
        with col2:
            amount_val = st.number_input("Amount", value=100.0, min_value=0.0, step=1.0)

        st.caption(
            "V1–V28 are PCA-anonymized features from the original dataset. "
            "Leave at 0.0 for a rough manual test, or paste real values from a test row."
        )
        v_cols = st.columns(4)
        v_values = {}
        for i, col_name in enumerate(config.FEATURE_COLUMNS_V):
            with v_cols[i % 4]:
                v_values[col_name] = st.number_input(col_name, value=0.0, format="%.4f", key=f"v_{i}")

        submitted = st.form_submit_button("Predict")

    if submitted:
        transaction = {config.TIME_COLUMN: time_val, config.AMOUNT_COLUMN: amount_val, **v_values}
        result = predict.predict_transaction(transaction)
        tier = risk_tier(result["fraud_probability"])

        st.markdown("### Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
        m2.metric("Risk tier", f"{TIER_COLOR[tier]} {tier}")
        m3.metric("Flagged as fraud", "Yes" if result["is_fraud"] else "No")

        with st.expander("Why? (SHAP explanation)"):
            try:
                from src import explain

                row_df = predict.preprocess_for_inference(pd.DataFrame([transaction]))
                contrib = explain.explain_single_prediction(row_df)
                st.dataframe(contrib.head(10), use_container_width=True)
                st.caption(
                    "Top features by absolute SHAP impact. Positive shap_value pushes "
                    "the prediction toward fraud; negative pushes toward legitimate."
                )
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

# ---------------------------------------------------------------------
# Page: Batch Upload
# ---------------------------------------------------------------------
elif page == "Batch Upload":
    require_artifacts()
    st.subheader("Score a batch of transactions")
    st.caption(
        f"CSV must contain columns: {', '.join([config.TIME_COLUMN] + config.FEATURE_COLUMNS_V + [config.AMOUNT_COLUMN])}"
    )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing_cols = set(config.ALL_FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {sorted(missing_cols)}")
        else:
            result = predict.predict_batch(df)
            result["risk_tier"] = result["fraud_probability"].apply(risk_tier)

            n_fraud = int(result["is_fraud"].sum())
            st.success(f"Scored {len(result)} transactions — {n_fraud} flagged as fraud.")

            tier_counts = result["risk_tier"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Low risk", int(tier_counts["Low"]))
            c2.metric("🟡 Medium risk", int(tier_counts["Medium"]))
            c3.metric("🔴 High risk", int(tier_counts["High"]))

            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download scored CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="fraudshield_scored_transactions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------
# Page: Live Simulator
# ---------------------------------------------------------------------
elif page == "Live Simulator":
    require_artifacts()
    st.subheader("Live transaction simulator")
    st.caption(
        "Streams transactions from the held-out test set (labels available for "
        "demo purposes only — the model never sees the label)."
    )

    n = st.slider("Number of transactions to stream", min_value=5, max_value=100, value=20)
    delay = st.slider("Delay between transactions (seconds)", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

    if st.button("Start simulation"):
        table_placeholder = st.empty()
        rows = []
        try:
            for result in simulator.stream_transactions(n=n, delay=delay):
                tier = risk_tier(result["fraud_probability"])
                rows.append(
                    {
                        "timestamp": result["timestamp"],
                        "amount": round(result["transaction"].get(config.AMOUNT_COLUMN, 0.0), 2),
                        "fraud_probability": result["fraud_probability"],
                        "risk_tier": f"{TIER_COLOR[tier]} {tier}",
                        "alert": "🚨" if result["alert"] else "",
                        "true_label": result["true_label"],
                    }
                )
                table_placeholder.dataframe(
                    pd.DataFrame(rows).iloc[::-1], use_container_width=True
                )
        except FileNotFoundError:
            st.error(
                f"No processed test set found at `{config.TEST_DATA_PATH}`. "
                "Run `python -m src.preprocess` first."
            )

# ---------------------------------------------------------------------
# Page: Model Insights
# ---------------------------------------------------------------------
elif page == "Model Insights":
    require_artifacts()
    st.subheader("Global feature importance (SHAP)")
    st.caption(
        "Summarizes, across a sample of the test set, which features push "
        "predictions toward fraud vs. legitimate on average."
    )

    sample_size = st.slider("Sample size", min_value=50, max_value=2000, value=500, step=50)

    if st.button("Generate SHAP summary"):
        with st.spinner("Computing SHAP values..."):
            try:
                from src import explain

                save_path = explain.plot_global_summary(sample_size=sample_size)
                st.image(str(save_path), caption="SHAP global feature importance")
            except Exception as e:
                st.error(f"Could not generate SHAP summary: {e}")
