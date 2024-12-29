import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Extract specific frequency components
def extract_frequency_components(data, target_freq, fs=1.0, bandwidth=0.05):
    n = len(data)
    data = np.asarray(data)  # Convert to NumPy array
    fft_values = fft(data)
    freqs = fftfreq(n, d=1/fs)

    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    filtered_fft = np.zeros_like(fft_values)
    filtered_fft[mask] = fft_values[mask]

    return np.real(ifft(filtered_fft))

# Fetch stock data
ticker = "TTRAK.IS"
data = yf.download(ticker, start="2020-01-01", end="2024-03-01")

if not data.empty:
    # Handle missing values
    data = data.dropna()
    data = data.asfreq('D')
    data['Close'] = data['Close'].interpolate(method='time')

    # Normalize the Close price
    data['Normalized_Close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

    # Extract specific frequency components (e.g., seasonal component)
    target_frequency = 1/30  # Annual cycle
    data['Seasonal_Component'] = extract_frequency_components(data['Normalized_Close'], target_frequency)

    # Denormalize the seasonal component
    data['Seasonal_Component'] = (data['Seasonal_Component'] * data['Close'].std()) + data['Close'].mean()

    # Decompose time series into trend, seasonal, and residual components
    decomposition = seasonal_decompose(data['Close'], model='additive', period=30)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot decomposition
    decomposition.plot()
    plt.show()

    # Apply Exponential Smoothing (Holt-Winters)
    hw_model = ExponentialSmoothing(
        data['Close'], trend="add", seasonal="add", seasonal_periods=365
    ).fit()

    # Forecast future values
    future_days = 365
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
    forecast = hw_model.forecast(future_days)

    # Combine historical and forecasted data
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast}).set_index("Date")
    data2 = yf.download(ticker, start="2023-01-01", end="2024-12-29")
    # Plot historical and forecasted data
    plt.figure(figsize=(14, 7))
    plt.plot(data2.index, data2['Close'], label="Historical Prices", color='blue')
    plt.plot(data.index, trend, label="Trend", color='orange')
    plt.plot(data.index, seasonal, label="Seasonal Component", color='green')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label="Holt-Winters Forecast", linestyle='dashed', color='red')
    plt.title("Historical Prices, Decomposition, and Holt-Winters Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

else:
    print(f"No data available for ticker: {ticker}")