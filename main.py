# Necessary Library Imported
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import pywt
from sklearn.linear_model import LinearRegression

# Low Pass Filter Function
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Extract Target Freq. using FFT Function
def extract_frequency_components(data, target_freq, fs=1.0, bandwidth=0.05):
    n = len(data)
    fft_values = fft(data.to_numpy())
    freqs = fftfreq(n, d=1 / fs)

    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    filtered_fft = np.zeros_like(fft_values)
    filtered_fft[mask] = fft_values[mask]

    return np.real(ifft(filtered_fft))

# Enhanced DSP Implementation Function
def apply_dsp_methods(data, fs=1.0, lowpass_cutoff=0.1, target_freq=1/90, bandwidth=0.05):
    filtered_data = lowpass_filter(data, lowpass_cutoff, fs)
    extracted_frequency = extract_frequency_components(pd.Series(filtered_data), target_freq, fs, bandwidth)

    return {
        "filtered_data": filtered_data,
        "extracted_frequency": extracted_frequency
    }

# Wavelet Transform-based Noise Reduction
def wavelet_transform_prediction(data, wavelet='db4', level=4, prediction_days=90, threshold_factor=0.5):
    if level is None:
        level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = threshold_factor * np.std(coeffs[-1])  # Threshold based on noise level
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    # Reconstruct the denoised signal using IDWT
    denoised_data = pywt.waverec(coeffs, wavelet)

    # Prepare data for prediction (Linear Regression on denoised signal)
    X = np.arange(len(denoised_data)).reshape(-1, 1)  # Time as input
    y = denoised_data  # Denoised values as target

    model = LinearRegression()
    model.fit(X, y)

    # Prediction
    future_X = np.arange(len(denoised_data), len(denoised_data) + prediction_days).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    return denoised_data, future_predictions

# Fetching Data
ticker = "THYAO.IS"
data = yf.download(ticker, start="2020-01-01", end="2024-01-19")
data2 = yf.download(ticker, start="2020-01-01", end="2024-04-20")
future_days = 90

if not data.empty:
    data = data.dropna()
    data = data.asfreq('D')
    data['Close'] = data['Close'].interpolate(method='time')

    # Normalize the Close price
    data['Normalized_Close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

    # Apply DSP methods
    dsp_results = apply_dsp_methods(data['Normalized_Close'], fs=1.0, lowpass_cutoff=0.1)

    # Add DSP results to the dataframe
    data['Filtered_Close'] = dsp_results['filtered_data']
    data['Extracted_Frequency'] = dsp_results['extracted_frequency']

    # Decompose time series into 3 component: trend, seasonal, and residual components
    decomposition = seasonal_decompose(data['Close'], model='additive', period=180)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot Decomposition
    decomposition.plot()
    plt.show()
    # Apply Exponential Smoothing (Holt-Winters)
    hw_model = ExponentialSmoothing(
        data['Close'], trend="add", seasonal="add", seasonal_periods=90, use_boxcox=True
    ).fit(optimized=True)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
    forecast = hw_model.forecast(future_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast}).set_index("Date")

    # Apply Wavelet Transform Prediction
    denoised_data, wavelet_predictions = wavelet_transform_prediction(data['Close'], prediction_days=future_days)
    wavelet_future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
    estimation_interval = forecast_df['Forecast'][0] - wavelet_predictions[0]
    wavelet_predictions = wavelet_predictions + estimation_interval

    # Golden Cross and Death Cross Data and Calculation
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    data['Death_Cross'] = (data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label="Close Price", color='blue')
    plt.plot(data['SMA_50'], label="50-Day SMA", color='orange')
    plt.plot(data['SMA_200'], label="200-Day SMA", color='green')

    # Highlight Golden Cross and Death Cross points
    plt.scatter(data.index[data['Golden_Cross']], data['Close'][data['Golden_Cross']], label='Golden Cross', marker='^', color='gold', s=200, edgecolor='black', linewidth=1.5)
    plt.scatter(data.index[data['Death_Cross']], data['Close'][data['Death_Cross']], label='Death Cross', marker='v', color='red', s=200, edgecolor='black', linewidth=1.5)

    plt.title("Golden Cross and Death Cross Detection")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

    # Visualization The Prediction and Historical Data
    plt.figure(figsize=(14, 7))
    plt.plot(data2.index, data2['Close'], label="Current predicted interval", color='blue')
    plt.plot(data.index, data['Close'], label="Historical Prices (Extended)", color='black')
    plt.plot(data.index, trend, label="Trend", color='orange')
    plt.plot(data.index, seasonal, label="Seasonal Component", color='green')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label="Holt-Winters Forecast", linestyle='dashed', color='red')
    plt.plot(wavelet_future_dates, wavelet_predictions, label="Wavelet Predictions", linestyle='dotted', color='purple')
    plt.title("Historical Prices, Decomposition, Holt-Winters, and Wavelet Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xlim(pd.Timestamp("2023-04-01"), pd.Timestamp("2024-04-01"))
    plt.grid()
    plt.show()

else:
    print(f"No data available for ticker: {ticker}")
