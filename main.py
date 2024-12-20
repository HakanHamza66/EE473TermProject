import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np
from pandas.plotting import autocorrelation_plot
from numpy.polynomial.polynomial import Polynomial
# Step 1: Fetch stock data
ticker = "AMZN"  # Example: Apple Inc.
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Ensure data is not empty
if not data.empty:
    # Step 2: Handle missing values
    data = data.dropna()

    # Step 3: Normalize the Close price
    data['Normalized_Close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

    # Step 4: Apply Noise Reduction using a low-pass filter
    def low_pass_filter(data, cutoff=0.1, fs=1.0, order=5):
        b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        return filtfilt(b, a, data)

    data['Filtered_Close'] = low_pass_filter(data['Normalized_Close'])

    # Step 5: Plot the original and noise-reduced data
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Normalized_Close'], label="Normalized Close Price", alpha=0.6)
    plt.plot(data.index, data['Filtered_Close'], label="Filtered Close Price (Noise Reduced)", linewidth=2)
    plt.title(f"{ticker} Normalized and Noise-Reduced Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid()
    plt.show()

    # Step 6: Apply Fourier Transform
    normalized_close = data['Normalized_Close'].values
    n = len(normalized_close)  # Total number of points
    timestep = 1  # Sampling interval (1 day for daily data)

    fft_values = fft(normalized_close)  # Perform FFT
    frequencies = fftfreq(n, d=timestep)  # Frequency bins

    # Extract positive frequencies
    positive_freq_idx = np.where(frequencies > 0)
    frequencies = frequencies[positive_freq_idx]
    fft_magnitude = np.abs(fft_values[positive_freq_idx])

    # Step 7: Plot the frequency spectrum
    plt.figure(figsize=(14, 7))
    plt.plot(frequencies, fft_magnitude, color='orange', linewidth=1.5)
    plt.title("Frequency Spectrum of Normalized Close Prices")
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
    # Step 1: Calculate SMA_50 and SMA_200
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Step 2: Identify crossovers
    data['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    data['Death_Cross'] = (data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))

    # Step 3: Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label="Close Price", color='blue')
    plt.plot(data['SMA_50'], label="50-Day SMA", color='orange')
    plt.plot(data['SMA_200'], label="200-Day SMA", color='green')

    # Highlight Golden Cross and Death Cross points
    plt.scatter(data.index[data['Golden_Cross']], data['Close'][data['Golden_Cross']], label='Golden Cross', marker='^',
                color='gold', s=100)
    plt.scatter(data.index[data['Death_Cross']], data['Close'][data['Death_Cross']], label='Death Cross', marker='v',
                color='red', s=100)

    plt.title("Moving Averages/ Golden and Dead Crosses")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

    # Step 4: Print the detected points
    golden_cross_points = data[data['Golden_Cross']]
    death_cross_points = data[data['Death_Cross']]

    autocorrelation_plot(data['Normalized_Close'])
    plt.title("Autocorrelation of Normalized Close Prices")
    plt.show()
    # Step 1: Extract time and normalized close prices
    data['Days'] = np.arange(len(data))  # Create time index as days
    x = data['Days'].values
    y = data['Normalized_Close'].values

    # Step 2: Fit a polynomial regression model (degree can be adjusted)
    degree = 5 # Degree of the polynomial
    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)

    # Step 3: Generate future points for extrapolation
    future_days = 30 # Predict 30 days into the future
    x_future = np.arange(len(data) + future_days)
    y_future = poly(x_future)

    # Step 4: Plot original and extrapolated data
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Normalized_Close'], label="Normalized Close Prices")
    plt.plot(pd.date_range(data.index[-1], periods=future_days + 1, freq='D')[1:], y_future[len(data):],
             label="Extrapolated Future Prices", linestyle='dashed', color='red')
    plt.title("Stock Price Extrapolation using Polynomial Regression")
    plt.xlabel("Date")
    plt.ylabel("Normalized Close Prices")
    plt.legend()
    plt.grid()
    plt.show()

    # Print extrapolated future values
    future_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='D')[1:]
    predictions = pd.DataFrame({'Date': future_dates, 'Predicted_Normalized_Close': y_future[len(data):]})
    print(predictions)
else:
    print(f"No data available for ticker: {ticker}")