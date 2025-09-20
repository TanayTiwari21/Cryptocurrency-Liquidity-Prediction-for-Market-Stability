# Cryptocurrency-Liquidity-Prediction-for-Market-Stability
Predict cryptocurrency liquidity and detect potential liquidity crises using machine learning and Streamlit.
🚀 Project Overview

This project analyzes historical cryptocurrency data to predict liquidity levels and identify potential liquidity crises. Using features like price, market capitalization, trading volume, and volatility, a Random Forest model predicts the liquidity ratio for various cryptocurrencies. A Streamlit web application provides a user-friendly interface for real-time insights.

🛠 Features

Data Preprocessing: Cleans and prepares historical cryptocurrency data for modeling.

Feature Engineering: Includes moving averages and volatility calculations.

Machine Learning Models:

Linear Regression (baseline)

Random Forest Regressor (optimized via GridSearchCV)

Liquidity Crisis Detection: Flags low liquidity periods based on predicted values.

Interactive Web App:

Upload CSV files

Select cryptocurrencies

Visualize predicted liquidity over time

Download predictions with crisis flags

📊 Visualizations

Correlation heatmaps between key numeric features

Bitcoin price trends over time

Predicted liquidity line plots with crisis thresholds

🧰 Tech Stack

Python: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

Web App: Streamlit

📁 Repository Structure
├── data/                     # Sample datasets
├── models/                   # Trained models (e.g., liquidity_prediction_model.pkl)
├── notebooks/                # Jupyter notebooks for EDA and modeling
├── streamlit_app.py          # Streamlit app for liquidity prediction
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

⚡ How to Run

Clone the repository:

git clone <repository_url>
cd cryptocurrency-liquidity-prediction


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run streamlit_app.py


Upload your historical cryptocurrency dataset and start predicting liquidity.

📈 Future Improvements

Incorporate additional features such as social sentiment, on-chain metrics, and macroeconomic indicators

Implement real-time predictions using live market data

Extend model to multi-day forecasting for better crisis prediction

🔗 References

CoinGecko API Documentation

Scikit-learn: Machine Learning in Python – Pedregosa et al., 2011

Streamlit Documentation: https://docs.streamlit.io
