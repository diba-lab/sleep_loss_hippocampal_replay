#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:12:30 2019

@author: bapung
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.stats as stats
import scipy.ndimage as gf
from OsCheck import DataDirPath, figDirPath, RawDataPath
import readOpenEphys as openEphys
# import scipy.signal as sg
# import scipy.stats as stats
# from scipy.signal import hilbert
#from SpectralAnalysis import lfpSpectMaze

#import seaborn as sns


sourceDir = RawDataPath() + 'SleepDeprivation/Beatrice/PRE/2019-04-22_04-10-57/'   #PRE
sourceDir = RawDataPath() + 'SleepDeprivation/Beatrice/PRE/2019-04-22_07-33-15/'   #MAZE
sourceDir = RawDataPath() + 'SleepDeprivation/Beatrice/POST/2019-04-22_08-24-45/'   #POST
filename = sourceDir + '100_CH10.continuous'


data1 = openEphys.pack(sourceDir)
















#with open(filename, 'rb') as f:
#        # Read header info, file length, and number of records
##    header = readHeader(f)
#    header = {}

    # Read the data as a string
    # Remove newlines and redundant "header." prefixes
    # The result should be a series of "key = value" strings, separated
    # by semicolons.
#    header_string = f.read(1024).decode("utf-8").replace('\n','').replace('header.','')
#kwargs
#    # Parse each key = value string separately
#    for pair in header_string.split(';'):
#        if '=' in pair:
#            key, value = pair.split(' = ')
#            key = key.strip()
#            value = value.strip()
#
#            # Convert some values to numeric
#            if key in ['bitVolts', 'sampleRate']:
#                header[key] = float(value)
#            elif key in ['blockLength', 'bufferSize', 'header_bytes']:
#                header[key] = int(value)
#            else:
#                # Keep as string
#                header[key] = value



#data= openEphys.load(filename)
#
#lfp = data['data']
#eeg = lfp[0:-1:24]
#
#
#def butter_bandpass(lowcut, highcut, fs, order=5):
#    nyq = 0.5 * 1250
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = sg.butter(order, [low, high], btype='band')
#    return b, a
#
#
#def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = sg.lfilter(b, a, data)
#    return y
#
#lowcut = 140
#highcut = 250
#
#y = butter_bandpass_filter(eeg, lowcut, highcut, 1250, order=6)
#
#
##b, a = sg.butter(4, 100, 'low', analog=True)
##sos = sg.butter(3, 100, btype = 'bandpass', fs=1250, output='sos')
##yf = sg.sosfilt(sos,eegnrem1)
#
#
#
#f,t,spect = sg.spectrogram(eeg, 1250,nperseg=2048*4, noverlap=2048*3)
#spect = gf.gaussian_filter(spect, sigma= 1.2)
#
#plt.clf()
#plt.plot(eeg)
#plt.plot(y)
#plt.pcolormesh(t/3600, f[0:575], spect[0:575,:],cmap = 'hot', vmin=60, vmax=2550)
#with open(filename, 'rb') as f:
#        # Read header info, file length, and number of records
#        header = f.read(1024)

#file = loadContinuous(sourceDir)
#data = file.read(5)
