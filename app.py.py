import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("liquidity_prediction_model.pkl")
    return model

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚀 Cryptocurrency Liquidity Crisis Detection")
st.write("This app forecasts liquidity levels and detects potential liquidity crises for selected cryptocurrencies.")

# Upload dataset
uploaded_file = st.file_uploader("Upload historical crypto dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Assume dataset has "crypto", "date", and features + target
    if "crypto" not in df.columns:
        st.error("Dataset must contain a 'crypto' column to filter different cryptocurrencies.")
    else:
        # Select crypto
        crypto_list = df["crypto"].unique().tolist()
        selected_crypto = st.selectbox("Select Cryptocurrency", crypto_list)

        # Filter dataset
        df_crypto = df[df["crypto"] == selected_crypto].copy()
        st.subheader(f"Data for {selected_crypto}")
        st.write(df_crypto.head())

        # Drop non-feature columns (keeping only features used during training)
        feature_cols = [col for col in df_crypto.columns if col not in ["crypto", "date", "liquidity"]]
        X = df_crypto[feature_cols]

        # Predict
        predictions = model.predict(X)
        df_crypto["Predicted_Liquidity"] = predictions

        # Define liquidity crisis threshold
        threshold = df_crypto["Predicted_Liquidity"].quantile(0.1)  # bottom 10% liquidity = crisis
        df_crypto["Crisis_Flag"] = df_crypto["Predicted_Liquidity"] < threshold

        # Show crisis stats
        crisis_days = df_crypto["Crisis_Flag"].sum()
        st.subheader("Liquidity Crisis Insights")
        st.write(f"🔴 Crisis Days Detected: **{crisis_days}**")
        st.write(f"📉 Crisis Threshold (10% quantile): {threshold:.2f}")

        if crisis_days > 0:
            st.warning(f"⚠️ Liquidity crises detected for {selected_crypto}. Traders should manage risks.")
        else:
            st.success(f"✅ No liquidity crises detected for {selected_crypto}.")

        # Plot
        st.subheader("Predicted Liquidity Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_crypto["date"], df_crypto["Predicted_Liquidity"], label="Predicted Liquidity")
        ax.axhline(y=threshold, color="red", linestyle="--", label="Crisis Threshold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Liquidity")
        ax.legend()
        st.pyplot(fig)

        # Download results
        csv = df_crypto.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions with Crisis Flags", csv, "predictions_with_crisis.csv", "text/csv")
