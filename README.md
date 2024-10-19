# Time-Series-Forcasting

Here’s a suitable README for your project on GitHub:

---

# Airline Passenger Forecasting Using Hybrid Models

## Overview
This project demonstrates three hybrid models to forecast the number of airline passengers. The models combine both linear and nonlinear approaches to predict future passenger numbers, aiming to capture both trend and seasonal patterns in the time series data.

The project includes:

1. **Additive Hybrid Model**: Combines a linear regression model with an LSTM (Long Short-Term Memory) model. The linear model forecasts are subtracted from the data to isolate the nonlinear residuals, which are modeled using LSTM. The final forecast is obtained by adding the linear and LSTM forecasts.

2. **Multiplicative Hybrid Model**: Similar to the additive model, but here, the linear model forecasts are divided from the data, and the residuals are modeled using LSTM. The final forecast is the product of the linear and LSTM model forecasts.

3. **STL Decomposition Based Hybrid Model**: Decomposes the time series into trend, seasonal, and residual components using STL decomposition. The trend is modeled with linear regression, the seasonal component with LSTM, and the residual with GRU (Gated Recurrent Units).

---

## Dataset
The dataset used is the airline passenger data, where the number of passengers traveling each month is recorded. It is loaded from a CSV file named `Passengers.csv` with columns:
- **Month**: Date of travel (YYYY-MM format).
- **#Passengers**: Number of passengers.

---

## Project Structure

- **Additive Hybrid Model** (`additive_model.py`): 
  - Linear Regression for modeling the trend component.
  - LSTM for modeling the nonlinear residuals.
  
- **Multiplicative Hybrid Model** (`multiplicative_model.py`):
  - Linear Regression for the trend component.
  - LSTM for nonlinear residual modeling with a multiplicative approach.
  
- **STL Decomposition Hybrid Model** (`stl_decomposition_model.py`):
  - STL Decomposition for separating trend, seasonal, and residual components.
  - Linear Regression for trend, LSTM for seasonal, and GRU for residual components.

- **Evaluation Metrics**: 
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - Accuracy Percentage
  
---

## How to Run

### Prerequisites
- Python 3.7+
- Required packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow`
  - `matplotlib`
  - `statsmodels`

Install dependencies:

pip install -r requirements.txt


### Running the Models
To run the models, execute the respective Python files:

1. **Additive Hybrid Model**:

   python additive_model.py
   

2. **Multiplicative Hybrid Model**:
   
   python multiplicative_model.py
  

3. **STL Decomposition Hybrid Model**:

   python stl_decomposition_model.py
  

---

## Results & Visualization

Each model generates a plot comparing the actual number of passengers with the predicted values (from the linear, LSTM, and hybrid models). Key evaluation metrics such as MAE, MSE, RMSE, and MAPE are printed to the console.

---

## Future Work
- Incorporate more advanced deep learning models like GRU or Transformer-based models.
- Experiment with different time series decomposition techniques.
- Extend the models for other types of time series data.
