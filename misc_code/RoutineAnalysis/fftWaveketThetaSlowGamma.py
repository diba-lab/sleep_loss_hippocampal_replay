import numpy as np
import matplotlib.pyplot as plt

# import pywt
import pandas as pd
import numpy as np
import scipy.signal as sg
import scipy.stats as stats
from sklearn.cluster import AgglomerativeClustering
import os


basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/"
)


subname = os.path.basename(os.path.normpath(basePath))
fileName = basePath + subname + ".eeg"
nChans = 134

reqChan = 33
b1 = np.memmap(fileName, dtype="int16", mode="r")
ThetaExtract = b1[reqChan::nChans]
# np.save(basePath+subname+'_Chan1.npy', ThetaExtract)


reqChan2 = 2
ThetaExtract2 = b1[reqChan2::nChans]
# np.save(basePath+subname+'_Chan2.npy', ThetaExtract2)


thetaData = ThetaExtract
sampleRate = 1250
N = len(thetaData)

corr1 = []
for ind in range(0, N, 5 * sampleRate):
    s1 = ThetaExtract[ind : ind + 10 * sampleRate]
    s2 = ThetaExtract2[ind : ind + 10 * sampleRate]
    a = np.corrcoef(s1, s2)
    corr1.append(a[0, 1])

# N = 500
# t = np.linspace(0, 1, N)
# s = 0.7*np.sin(8 * 2 * np.pi * t)+0.5*np.sin(30 * 2 * np.pi * t) + 0.3 * np.sin(60 * 2 * np.pi * t)+0.2*np.random.randn(500)
# dt = t[1]-t[0]

# fftSig = np.abs(np.fft.fft(thetaData))
# freq = np.fft.fftfreq(N, 1/sampleRate)

f, t, sxx = sg.spectrogram(
    thetaData, fs=sampleRate, nperseg=10 * sampleRate, noverlap=5 * sampleRate
)

# f2, coher_sig = sg.coherence(ThetaExtract, ThetaExtract2, fs=1250,
#                              nperseg=10*sampleRate, noverlap=5*sampleRate)

theta_ind = np.where((f > 5) & (f < 10))[0]
delta_ind = np.where(((f > 1) & (f < 4)) | ((f > 12) & (f < 15)))[
    0
]  # delta band 0-4 Hz and 12-15 Hz
gamma_ind = np.where((f > 50) & (f < 250))[0]  # delta band 0-4 Hz and 12-15 Hz
highfreq_ind = np.where((f > 300) & (f < 600))[0]  # delta band 0-4 Hz and 12-15 Hz

theta_sxx = np.mean(sxx[theta_ind, :], axis=0)
delta_sxx = np.mean(sxx[delta_ind, :], axis=0)
gamma_sxx = np.mean(sxx[gamma_ind, :], axis=0)
highfreq_sxx = np.mean(sxx[highfreq_ind, :], axis=0)

theta_delta_ratio = stats.zscore(theta_sxx / delta_sxx)
theta_gamma_ratio = stats.zscore(theta_sxx / gamma_sxx)
theta_highfreq_ratio = stats.zscore(theta_sxx / highfreq_sxx)

feature_comb = np.stack((theta_delta_ratio, theta_gamma_ratio), axis=1)

# cluster = AgglomerativeClustering(
#     n_clusters=2, affinity='euclidean', linkage='ward')
# cluster.fit_predict(feature_comb)


# TODO coherence among multiple channels from different shanks
theta_ratio_dist, bin_edges = np.histogram(theta_delta_ratio, bins=100)

plt.clf()
plt.plot(theta_highfreq_ratio, theta_delta_ratio, ".")
# plt.plot(freq[:N//2], (2/N)*fftSig[:N//2])
# plt.imshow(np.flip(sxx), cmap="jet", vmax=5000, extent=[0, 14, 0, 1])
# plt.plot(theta_delta_ratio)
