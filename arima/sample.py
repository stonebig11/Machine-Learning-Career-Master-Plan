import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate synthetic time series data
np.random.seed(0)
time = np.arange(1, 101)
data = 0.5 * time + 2 * np.sin(0.1 * time) + np.random.normal(0, 1, 100)

# Create a pandas DataFrame with the time series data
ts = pd.Series(data, index=pd.date_range('20230101', periods=100))

# Define the ARIMA order (p, d, q)
p = 2  # Autoregressive order
d = 1  # Differencing order
q = 1  # Moving Average order

# Create and fit an ARIMA model
model = ARIMA(ts, order=(p, d, q))
model_fit = model.fit()

# Make predictions
n = 10  # Number of steps to forecast
predictions = model_fit.forecast(steps=n)
# Create an index for the future time steps
future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(1, unit='D'), periods=n)


# Visualize the original data and forecasts
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original Data')
plt.plot(future_dates, predictions, label='ARIMA Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.title('ARIMA Time Series Forecasting')
plt.show()