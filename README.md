# Forecasting-System Using Different Time Series Models
Overview
This project implements and compares different time series forecasting models: ARIMA (AutoRegressive Integrated Moving Average), ANN (Artificial Neural Network), and a hybrid ARIMA-ANN model. The primary goal is to analyze their performance on a time series dataset and determine which model provides the most accurate forecasts.
## Introduction
Time series forecasting is a critical task in various domains, including finance, weather forecasting, and sales prediction. This project explores the effectiveness of traditional statistical methods (ARIMA, SARIMA, ETS), machine learning approaches (ANN, LSTM, SVR, Prophet), and hybrid models (ARIMA-ANN) in forecasting time series data.

## Features
Implementation of ARIMA model
Implementation of ANN model using a neural network
Development of a hybrid ARIMA-ANN model
Implementation of SARIMA model
Implementation of Prophet model
Implementation of Support Vector Regression (SVR) model
Implementation of Long Short-Term Memory (LSTM) model
Implementation of Exponential Smoothing (ETS) Model
Comparative analysis of the models
Visualization of forecasting results
## Installation
To run this project, you need to have Python installed on your system. You can install the necessary packages using the following command:
pip install -r requirements.txt
The requirements.txt file should include the following packages:
numpy
pandas
matplotlib
statsmodels
scikit-learn
keras
tensorflow
fbprophet
scipy
## Models Implemented
ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a popular statistical method for time series forecasting that uses past values and past forecast errors to predict future values.

SARIMA (Seasonal ARIMA)
SARIMA extends ARIMA by explicitly supporting univariate time series data with a seasonal component.

ANN (Artificial Neural Network)
ANN is a machine learning approach that mimics the human brain's neural networks. It can capture non-linear relationships in the data, making it suitable for complex time series forecasting.

LSTM (Long Short-Term Memory)
LSTM is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies, making it highly effective for sequential data and time series forecasting.

Prophet
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

SVR (Support Vector Regression)
SVR is a type of Support Vector Machine (SVM) that is used for regression challenges. It is effective in high-dimensional spaces and is robust to outliers.

ETS (Exponential Smoothing)
Exponential Smoothing models are forecasting models that account for level, trend, and seasonality in time series data.

Hybrid ARIMA-ANN
The hybrid model combines the strengths of both ARIMA and ANN. It first applies ARIMA to capture the linear components of the time series and then uses ANN to model the non-linear residuals.

## Dataset
The dataset used for this project is a time series dataset. Ensure that your data is preprocessed and formatted correctly. An example dataset can be placed in the data directory.

## Results
The results of the model comparisons, including performance metrics and visualizations, will be displayed after running the main script. Detailed results and analysis can be found in the results directory.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
