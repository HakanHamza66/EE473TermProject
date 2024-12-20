import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np

# Step 1: Fetch stock data
ticker = "AAPL"  # Example: Apple Inc.
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

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

else:
    print(f"No data available for ticker: {ticker}")