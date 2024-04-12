import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.stats as stat
from hmmlearn.hmm import GaussianHMM

basePath = (
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-05-31_03-55-36/"
)
SampFreq = 1250
nyq = 0.5 * 1250

sessionName = os.path.basename(os.path.normpath(basePath))
filename = basePath + sessionName + "_BestRippleChans.npy"

data = np.load(filename, allow_pickle=True)


signal = data.item()
signal = signal["BestChan"]

signal = signal[::125]

b, a = sg.butter(3, [150 / nyq, 250 / nyq], btype="bandpass")
yf = sg.filtfilt(b, a, signal)

squared_signal = np.square(yf)
normsquaredsignal = stat.zscore(squared_signal)

# getting an envelope of the signal
# analytic_signal = sg.hilbert(yf)
# amplitude_envelope = stat.zscore(np.abs(analytic_signal))

windowLength = SampFreq / SampFreq * 11
window = np.ones((int(windowLength),)) / windowLength

smoothSignal = sg.filtfilt(window, 1, squared_signal, axis=0)
zscoreSignal = stat.zscore(smoothSignal)

hist_zscore, edges = np.histogram(zscoreSignal, bins=200)


sig_zscore = stat.zscore(signal)


signal = np.reshape(sig_zscore, [len(sig_zscore), 1])
yf = np.reshape(squared_signal, [len(squared_signal), 1])

# signal = signal[0 : 1250 * 1000]
model = GaussianHMM(n_components=2, n_iter=10).fit(signal)
hidden_states_noise = model.predict(signal)

# yf = yf[: 1250 * 1000]
# model = GaussianHMM(n_components=2, n_iter=100).fit(yf)
# hidden_states_ripple = model.predict(yf)


plt.clf()
plt.subplot(211)
plt.plot(signal)
plt.plot(hidden_states_noise)

# plt.subplot(212)
# plt.plot(yf)
# plt.plot(hidden_states_ripple)

# plt.subplot(311)
# plt.plot(signal)

# plt.subplot(312)
# plt.plot(yf)

# plt.subplot(313)
# plt.plot(sig_zscore, "r")
# plt.plot(zscoreSignal)

