"""
Created on Sat Apr 22 20:29:22 2023

@author: ashristi27@gmail.com
"""
from pylab import *
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import filtfilt

# Load data from CSV file
data = pd.read_csv('Unprocessed_Test_8.csv')
len_data = len(data)
time = data['time'].values
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

# Compute the frequency axis
freq = np.fft.fftfreq(len_data, 1 / fs)

# Find maximum peak in each signal and denote by a blue cross
max_freq_amp = []
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 7.19))
for i in range(fft_signals.shape[1]):
    ax = axs[i]
    ax.plot(freq, np.abs(fft_signals[:, i]), linewidth=0.25,
            label=data.columns[fft_signals.shape[1] - i], color='red')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 1200])
    ax.set_title(f'Signal {fft_signals.shape[1] - i}: {data.columns[fft_signals.shape[1] - i]}')
    ax.legend()

    # Find the maximum peak and denote by a blue cross
    max_index = np.argmax(np.abs(fft_signals[:, i]))
    max_freq = freq[max_index]
    max_amp = np.abs(fft_signals[max_index, i])
    max_freq_amp.append((max_freq, max_amp))
    ax.plot(max_freq, max_amp, 'bx', markersize=10)

# Print frequency and amplitude at the maximum peak of each signal
for i, (freq, amp) in enumerate(max_freq_amp):
    print(f"Signal {i + 1}-Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}")

# Set title
fig.suptitle('FFT of Ground Motion Data 8')
plt.tight_layout()
plt.show()

# Save plot as JPG file
plt.savefig('FFT of Ground Motion Data 8.jpg', dpi=400)
