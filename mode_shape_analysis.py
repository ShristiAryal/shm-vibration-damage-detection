"""
Created on Thu Apr 20 14:28:33 2023

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
data = pd.read_csv('Unprocessed_Test_9.csv')
len_data = len(data)
time = data['time'].values
signals = data[['Acc_5', 'Acc_9', 'Acc_13', 'Acc_17']].values

# Define time range
start_time = 10
end_time = 50

# Set up the figure and axes
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharex=True)

# Find the indices corresponding to the boundary start and end times
start_idx = np.where(time >= start_time)[0][0]
end_idx = np.where(time <= end_time)[-1][-1]

# Apply a Butterworth bandpass filter
fs = 1024       # Sampling rate (Hz)
fc = 7.28       # Cutoff frequency (Hz)
order = 2       # Filter order
nyq = 0.5 * fs  # Nyquist frequency
Wn = [(fc - 0.01), (fc + 0.01)]  # Normalized cutoff frequency
b, a = butter(order, Wn, fs=fs, btype='band')
signals_filt = filtfilt(b, a, signals, axis=0)

# Find the indices corresponding to the start and end times of the time range
start_idx_range = np.where(time[start_idx:end_idx] >= start_time)[0][0] + start_idx
end_idx_range = np.where(time[start_idx:end_idx] <= end_time)[-1][-1] + start_idx

# Plot filtered signals within the time range of interest
ax.plot(time[start_idx_range:end_idx_range],
        signals_filt[start_idx_range:end_idx_range, 3],
        label='Acc_17', color='green')
ax.plot(time[start_idx_range:end_idx_range],
        signals_filt[start_idx_range:end_idx_range, 2],
        label='Acc_13', color='orange')
ax.plot(time[start_idx_range:end_idx_range],
        signals_filt[start_idx_range:end_idx_range, 1],
        label='Acc_9', color='blue')
ax.plot(time[start_idx_range:end_idx_range],
        signals_filt[start_idx_range:end_idx_range, 0],
        label='Acc_5', color='red')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration')
ax.set_title('Filtered Signals')
ax.legend()

f2 = plt.figure()

# Define empty lists to store maximum displacement and its corresponding time
max_displacement_filt = []
max_displacement_filt_times = []

# Compute velocity by integrating acceleration using cumtrapz
dt = 1 / fs
velocity = np.zeros_like(signals_filt)
velocity_filt = np.zeros_like(signals_filt)
for i in range(signals_filt.shape[1]):
    velocity[:, i] = scipy.integrate.cumtrapz(signals_filt[:, i], dx=dt, initial=0)
    velocity_filt[:, i] = filtfilt(b, a, velocity[:, i], axis=0)

# Compute displacement by integrating velocity_filt using cumtrapz
dt = 1 / fs
displacement = np.zeros_like(velocity_filt)
displacement_filt = np.zeros_like(velocity_filt)
for i in range(velocity_filt.shape[1]):
    displacement[:, i] = scipy.integrate.cumtrapz(velocity_filt[:, i], dx=dt, initial=0)
    displacement_filt[:, i] = filtfilt(b, a, displacement[:, i], axis=0)

# Compute maximum displacement for each signal
max_disp = []
for i in range(displacement_filt.shape[1]):
    max_disp.append(np.max(np.abs(displacement_filt[:, i])))

# Normalize maximum displacement
max_displacement_norm = max_disp / max(max_disp)

# Create a numpy array from the two lists
max_displacement_norm_times_array = np.column_stack((max_displacement_norm, [1, 2, 3, 4]))

# Print the array
print('Maximum Displacements and Times:\n', max_displacement_norm_times_array)

# Create a plot of the normalized maximum displacements
plt.plot(max_displacement_norm, [1, 2, 3, 4])

plt.title('Normalized Maximum Displacements vs Floor')
plt.xlabel('Normalized Maximum Displacement')
plt.ylabel('Floor')

plt.show()

# Save plot as JPG file
plt.savefig('Unprocessed_Test_9.jpg', dpi=400)
