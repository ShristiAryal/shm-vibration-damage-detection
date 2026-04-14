"""
Created on Sat Apr 22 22:22:17 2023

@author: ashristi27@gmail.com
"""
from pylab import *
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, csd
from scipy.signal import filtfilt

# Load data from CSV file
data = pd.read_csv('Unprocessed_Test_9.csv')
len_data = len(data)
time = data['time'].values
# Reverse the order of the columns
signals = data[['Acc_5', 'Acc_9', 'Acc_13', 'Acc_17']].iloc[:, ::-1].values

# Define filter parameters
lowcut = 1   # Hz
highcut = 50  # Hz
order = 2    # Order of the filter

# Define sampling frequency
fs = 1024  # Hz

# Baseline correction
for i in range(signals.shape[1]):
    signals[:, i] = signals[:, i] - np.mean(signals[:, i])

# Apply bandpass filter to each signal
b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
filtered_signals = np.zeros_like(signals)
for i in range(signals.shape[1]):
    filtered_signals[:, i] = filtfilt(b, a, signals[:, i])

# Compute the FFT of the filtered signals
fft_signals = np.zeros_like(filtered_signals)
for i in range(filtered_signals.shape[1]):
    fft_signals[:, i] = np.fft.fft(filtered_signals[:, i])

# Compute the cross-spectral density for each pair of signals
csd_signals = np.zeros((len_data // 2 + 1, filtered_signals.shape[1],
                         filtered_signals.shape[1]), dtype=np.complex128)
for i in range(filtered_signals.shape[1]):
    for j in range(i, filtered_signals.shape[1]):
        f, csd_signals[:, i, j] = scipy.signal.csd(
            filtered_signals[:, i],
            filtered_signals[:, j],
            fs=fs,
            window='hann',
            nperseg=1024,
            noverlap=128,
            nfft=len_data
        )
        csd_signals[:, j, i] = csd_signals[:, i, j].conjugate()

# Compute the frequencies array
frequencies = np.fft.fftfreq(len_data, 1 / fs)[:len_data // 2 + 1]

# Compute the PSD for each signal
psd_signals = np.zeros((len_data // 2 + 1, filtered_signals.shape[1]))
for i in range(filtered_signals.shape[1]):
    # Compute the magnitude of the Fourier coefficients
    magnitude = np.abs(fft_signals[:len_data // 2 + 1, i])
    # Square the magnitude values to obtain the power spectrum
    power_spectrum = magnitude ** 2
    # Compute the PSD
    psd_signals[:, i] = power_spectrum / (len(filtered_signals) * fs)

# Plot the PSD of each signal in a separate subplot
fig, axes = plt.subplots(nrows=filtered_signals.shape[1], ncols=1, figsize=(8, 7))
for i in range(filtered_signals.shape[1]):
    axes[i].plot(frequencies, psd_signals[:, i], linewidth=0.25,
                 label=data.columns[fft_signals.shape[1] - i], color='red')
    axes[i].set_xlabel('Frequency (Hz)')
    axes[i].set_ylabel('PSD')
    axes[i].set_title('Signal {}'.format(i + 1))
    axes[i].legend()
    axes[i].set_xlim([0, 10])
    axes[i].set_ylim([0, 0.02])

fig.suptitle('PSD of Ground Motion Data 9')
plt.tight_layout()
plt.show()

# Loop over the signals and print max peak frequency
for i in range(filtered_signals.shape[1]):
    max_peak_index = np.argmax(psd_signals[:, i])
    max_peak_freq = frequencies[max_peak_index]
    print("Max peak frequency for signal", i + 1, ":", max_peak_freq)

# Save plot as JPG file
plt.savefig('PSD for Ground Motion Data 9.jpg', dpi=400)
