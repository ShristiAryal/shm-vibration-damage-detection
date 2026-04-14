# Vibration-Based Damage Detection in RC Structures

This repository contains the Python signal processing pipeline developed for my Master's Special Project at Bauhaus-Universität Weimar. 

The codebase was written to process raw, high-frequency accelerometer data from shake-table tests to identify structural damage using output only methods.

## Data Availability Statement 
Please note: The raw shake-table experimental data (CSV files) utilized in this project belongs to the testing facility. Therefore, the dataset cannot be shared publicly in this repository. The Python scripts provided here are meant to demonstrate the algorithmic logic, methodology, and pipeline  developed for the analysis.

## Files in this Repository
* `fft_analysis.py`: Applies baseline correction, Butterworth bandpass filtering, and computes the Fast Fourier Transform (FFT) to extract dominant frequencies.
* `mode_shape_analysis.py`: Uses cumulative trapezoidal integration to convert acceleration to displacement and identifies mode shape curvature changes (plastic hinges).
* `fdd_analysis.py`: Performs Frequency Domain Decomposition (FDD) using Singular Value Decomposition (SVD) on the PSD matrix.
* `cwt_spectrogram.py`: Applies the Morlet Continuous Wavelet Transform (CWT) to generate time-frequency spectrograms of transient damage events.

## Dependencies
This pipeline was built using standard scientific Python libraries:
* `numpy`
* `scipy`
* `matplotlib`
* `pandas`
* `pywt` (PyWavelets)
