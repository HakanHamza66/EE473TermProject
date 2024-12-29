import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np
from pandas.plotting import autocorrelation_plot
from scipy.signal import freqz
import pywt

# Step 1: Fetch stock data
ticker = "TTRAK.IS"  # Example: Apple Inc.
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

    # Step 8: Generate Predictions and Compare with Real Data
    # Step 1: Extract time and normalized close prices
    data['Days'] = np.arange(len(data))  # Create time index as days
    x = data['Days'].values
    y = data['Normalized_Close'].values

    # Step 2: Fit a polynomial regression model (degree can be adjusted)
    degree = 5
    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)

    # Step 3: Generate future points for extrapolation
    future_days = 300 # Predict 90 days into the future
    x_future = np.arange(len(data) + future_days)
    # Extract the last known value
    last_original_value = data['Normalized_Close'].iloc[-1]

    # Generate raw predictions
    y_future_raw = poly(x_future)

    # Align future predictions with the last original value
    y_future = y_future_raw + (last_original_value - y_future_raw[len(data)])

    # Create a future date range
    future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq='D')[1:]

    # Combine original and predicted data
    all_dates = data.index.append(future_dates)
    all_normalized_values = np.concatenate([data['Normalized_Close'].values, y_future[len(data):]])

    # Fetch actual data for the prediction range
    actual_future_data = yf.download(ticker, start=future_dates[0].strftime('%Y-%m-%d'), end=future_dates[-1].strftime('%Y-%m-%d'))

    if not actual_future_data.empty:
        actual_future_data['Normalized_Close'] = (actual_future_data['Close'] - data['Close'].mean()) / data['Close'].std()

        # Plot the historical, predicted, and actual future data
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Normalized_Close'], label="Historical Normalized Close Prices")
        plt.plot(future_dates, y_future[len(data):], label="Extrapolated Future Prices", linestyle='dashed', color='red')
        plt.plot(actual_future_data.index, actual_future_data['Normalized_Close'], label="Actual Future Prices", color='green')
        plt.title("Stock Price Extrapolation with Real Data Comparison")
        plt.xlabel("Date")
        plt.ylabel("Normalized Close Prices")
        plt.legend()
        plt.grid()
        plt.show()
        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Normalized_Close': y_future[len(data):]
        })
        predictions.set_index('Date', inplace=True)
        predictions['Actual_Normalized_Close'] = actual_future_data['Normalized_Close'].reindex(predictions.index)

        print(predictions)
    else:
        print("No actual data available for the prediction range.")

    # Step 9: Wavelet Transform for Future Prediction
    def wavelet_transform_prediction(data, wavelet='db4', level=4, future_days=90):
        # Step 1: Perform Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec(data, wavelet, level=level)

        # Reconstruct the trend from approximation coefficients
        reconstructed_trend = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)

        # Extend trend for future prediction
        x = np.arange(len(reconstructed_trend))
        future_x = np.arange(len(reconstructed_trend) + future_days)

        # Fit a polynomial model to the trend
        poly_coefs = np.polyfit(x, reconstructed_trend, 3)  # Degree can be adjusted
        poly_model = np.poly1d(poly_coefs)

        # Predict future trend
        future_trend = poly_model(future_x)

        return reconstructed_trend, future_trend[-future_days:]  # Return both reconstructed trend and future predictions

    # Perform Wavelet Transform Prediction
    reconstructed_trend, future_trend = wavelet_transform_prediction(normalized_close, future_days=future_days)

    # Step 10: Generate Future Dates and Combine Results
    future_dates_wavelet = pd.date_range(start=data.index[-1], periods=len(future_trend) + 1, freq='D')[1:]

    # Fetch actual data for the prediction range (Wavelet)
    actual_wavelet_future_data = yf.download(ticker, start=future_dates_wavelet[0].strftime('%Y-%m-%d'), end=future_dates_wavelet[-1].strftime('%Y-%m-%d'))

    if not actual_wavelet_future_data.empty:
        actual_wavelet_future_data['Normalized_Close'] = (actual_wavelet_future_data['Close'] - data['Close'].mean()) / data['Close'].std()

        # Plot Historical Data with Wavelet Prediction and Actual Data
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, reconstructed_trend[:len(data)], label="Reconstructed Trend (Wavelet)")
        plt.plot(data.index, data['Normalized_Close'], label="Historical Normalized Close Prices", alpha=0.6)
        plt.plot(future_dates_wavelet, future_trend, label="Wavelet Transform Prediction", linestyle='dotted', color='purple')
        plt.plot(actual_wavelet_future_data.index, actual_wavelet_future_data['Normalized_Close'], label="Actual Future Prices (Wavelet)", color='orange')
        plt.title("Stock Price Prediction using Wavelet Transform with Real Data")
        plt.xlabel("Date")
        plt.ylabel("Normalized Close Prices")
        plt.legend()
        plt.grid()
        plt.show()

        # Combine Wavelet Predictions with Actual Data for Comparison
        wavelet_predictions = pd.DataFrame({
            'Date': future_dates_wavelet,
            'Wavelet_Predicted_Normalized_Close': future_trend
        })
        wavelet_predictions.set_index('Date', inplace=True)
        wavelet_predictions['Actual_Normalized_Close'] = actual_wavelet_future_data['Normalized_Close'].reindex(wavelet_predictions.index)

        print(wavelet_predictions)
    else:
        print("No actual data available for the wavelet prediction range.")

else:
    print(f"No data available for ticker: {ticker}")
