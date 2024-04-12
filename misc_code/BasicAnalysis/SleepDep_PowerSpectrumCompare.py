#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
# import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28


offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2 + 75*2*SampFreq*3600*10, shape=(1, nChans * SampFreq * 3600 * 1))
eegnrem1 = b1[0, ::nChans]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
yf = ft.fft(eegnrem1) / len(eegnrem1)
xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
y1 = smth.gaussian_filter(y1, 200)

plt.clf()
# plt.subplot()
plt.plot(xf, y1)
plt.xlim([1, 80])
plt.ylim([0, 0.000005])
plt.xlabel('Frequency')
plt.ylabel('Power')

filename = '/data/Clustering/SleepDeprivation/RatJ/Day2/RatJ_2019-06-02_03-59-19/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'


SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 67
ReqChan = 61


offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2 + 75*2*SampFreq*3600*5, shape=(1, nChans * SampFreq * 3600 * 1))
eegnrem1 = b1[0, ::nChans]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
yf = ft.fft(eegnrem1) / len(eegnrem1)
xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
y1 = smth.gaussian_filter(y1, 200)


# plt.subplot()
plt.plot(xf, y1)
plt.xlim([1, 80])
plt.ylim([0, 0.000005])
plt.xlabel('Frequency')
plt.ylabel('Power')
