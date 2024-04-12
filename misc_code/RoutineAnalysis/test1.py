import numpy as np
import matplotlib.pyplot as plt

# import pywt
import pandas as pd
import numpy as np
import scipy.signal as sg
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
import os

# from sklearn.cluster import AgglomerativeClustering


basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/"
)
SampFreq = 1250
nyq = 0.5 * 1250

sessionName = os.path.basename(os.path.normpath(basePath))
filename = basePath + sessionName + "_BestRippleChans.npy"
data = np.load(filename, allow_pickle=True)
signal = data.item()
thetaData = signal["BestChan"]


sampleRate = 1250
N = len(thetaData)

# N = 500
# t = np.linspace(0, 1, N)
# s = 0.7*np.sin(8 * 2 * np.pi * t)+0.5*np.sin(30 * 2 * np.pi * t) + 0.3 * np.sin(60 * 2 * np.pi * t)+0.2*np.random.randn(500)
# dt = t[1]-t[0]

# fftSig = np.abs(np.fft.fft(thetaData))
# freq = np.fft.fftfreq(N, 1/sampleRate)

f, t, sxx = sg.spectrogram(
    thetaData, fs=sampleRate, nperseg=10 * sampleRate, noverlap=5 * sampleRate
)

theta_ind = np.where((f > 5) & (f < 10))[0]
delta_ind = np.where((f < 4) | ((f > 12) & (f < 15)))[
    0
]  # delta band 0-4 Hz and 12-15 Hz
gamma_ind = np.where((f > 50) & (f < 250))[0]  # delta band 0-4 Hz and 12-15 Hz

theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)

theta_delta_ratio = stats.zscore(theta_sxx / delta_sxx)
theta_gamma_ratio = stats.zscore(theta_sxx / gamma_sxx)
theta_delta_ratio = np.reshape(theta_delta_ratio, [len(theta_delta_ratio), 1])

# feature_comb = np.stack ((theta_delta_ratio,theta_gamma_ratio),axis=1)

# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(feature_comb)
model = GaussianHMM(n_components=4, n_iter=100).fit(theta_delta_ratio)
hidden_states = model.predict(theta_delta_ratio)

# # TODO coherence among multiple channels from different shanks
# theta_ratio_dist, bin_edges = np.histogram(theta_delta_ratio,bins=100)

# plt.plot(theta_gamma_ratio,theta_delta_ratio,'.')
# plt.plot(freq[:N//2], (2/N)*fftSig[:N//2])

plt.clf()

plt.plot(theta_delta_ratio)
plt.plot(hidden_states)
