#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:32:41 2019

@author: bapung
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import scipy.ndimage as smth
import scipy.signal as sg
import scipy.stats as stat
import seaborn as sns

filename = '/data/Clustering/SleepDeprivation/RatJ/Day1/RatJ_2019-05-31_03-55-36/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.eeg'

SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=1 * (ReqChan - 1) * 2, shape=(1, nChans * SampFreq * 3600 * 15))
eegnrem1 = b1[0, ::nChans]
# eegnrem1 = eegnrem1[0::24]
# sos = sg.butter(3, 100, btype='low', fs=SampFreq, output='sos')
# yf = sg.sosfilt(sos, eegnrem1)
# yf = ft.fft(yf) / len(eegnrem1)
# xf = np.linspace(0.0, SampFreq / 2, len(eegnrem1) // 2)
# y1 = 2.0 / (len(xf)) * np.abs(yf[:len(eegnrem1) // 2])
# y1 = smth.gaussi
