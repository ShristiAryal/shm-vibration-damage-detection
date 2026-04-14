"""
Created on Mon Apr 24 21:59:21 2023

@author: ashristi27@gmail.com
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import cmath
import pywt

acc = pd.read_csv('Unprocessed_Test_4.csv')
F = 1 / acc['time'][1]
dt = acc['time'][1] - acc['time'][0]
signal_test = acc['Acc_5'][20480:35840]
time = acc['time'][20480:35840]

plt.figure()
plt.plot(time, signal_test)
plt.show()

min_scale = 1
max_scale = 100
num_scales = 200
scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales)

w = 6
C = 2
C = '%f' % C
B = w
B = '%f' % B

func = 'cmor' + B + '-' + C


def plot_wavelet(time, signal_test, scales, dt,
                 cmap=plt.cm.seismic,
                 title='Wavelet Transform(Power Spectrum) of signal',
                 ylabel='Period(s)',
                 xlabel='Time(s)'):
    cwtmatr, frequencies = pywt.cwt(signal_test.to_numpy(), scales, func, dt)
    power = (np.abs(cwtmatr)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, period, np.log2(power + 1e-12))

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()


plot_wavelet(time, signal_test, scales, dt)
plt.savefig('CWT of Ground Motion Data 4 Acc_5.jpg', dpi=400)
