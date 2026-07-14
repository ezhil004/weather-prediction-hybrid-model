# Weather Prediction: A Hybrid ARIMA - Random Forest Approach

This repository contains the codebase for the project "Weather Prediction: A Hybrid ARIMA - Random Forest Approach" by Tharun Venkat M, Vishwa . V, and EzhilVendhan.

## Overview
This project presents a hybrid approach that leverages:
- **Random Forest**, a robust machine learning algorithm, for multiclass weather condition prediction.
- **ARIMA** (AutoRegressive Integrated Moving Average) for time-series trend analysis of weather attributes such as temperature, precipitation, and wind speed.

## Files
- `train_model.py`: Script to preprocess data, train the Random Forest model on historical data, and save the trained model, scaler, and label encoder.
- `Weather.py`: A Streamlit web application that provides:
  - Real-time weather predictions by integrating with the OpenWeatherMap API.
  - Historical weather trends visualizations via the Visual Crossing API.
  - 7-Day weather forecasts combining ARIMA time-series predictions with the Random Forest model's weather classification.

## Setup
You will need to install the dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit requests statsmodels
```

Run the Streamlit app:
```bash
streamlit run Weather.py
```
*(Ensure `train_model.py` is run first to generate the `.pkl` files, and place the `weather_classification_data.csv` dataset in the root folder.)*
