import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.ndimage as smth
import scipy.stats as stat

# signal = np.load('/data/DataGen/SleepDeprivation/RatJDay1.npy')
filename = '/home/bapung/Documents/ClusteringHub/RatJ_2019-05-31_03-55-36/continuous.eeg'
SampFreq = 1250
#frames = RecInfo['behavFrames']
#behav = RecInfo['behav']
nChans = 75
ReqChan = 28

# offsetP = ((nREMPeriod - behav[2, 0]) // 1e6) * SampFreq + \
#    int(np.diff(frames[0, :])) + int(np.diff(frames[1, :]))

offsetp = (ReqChan-1)

b1 = np.memmap(filename, dtype='int16', mode='r',
               offset=0, shape=(SampFreq * 3600 * 3, nChans))
chan1 = np.array(b1[:, 25], dtype=np.float)
chan2 = np.array(b1[:, 5], dtype=np.float)


nyq = 0.5 * SampFreq

signal = chan1
signal = np.array(signal, dtype=np.float)  # convert data to float


f, t, Sxx = sg.spectrogram(
    signal, SampFreq, nperseg=1250 * 3, noverlap=1250 * 2, nfft=5000)


#zscoreSignal = stat.zscore(signal)

coher = []
for i in np.linspace(0, len(chan1), 10798).tolist():

    c1 = chan1[int(i):int(i)+3*1250]
    c2 = chan2[int(i):int(i)+3*1250]
    coher.append(np.mean(sg.coherence(c1, c2, fs=1250)[1]))
    # coher.append(np.correlate(c1, c2))


b, a = sg.butter(3, [5/nyq, 10/nyq], btype='bandpass')
theta = sg.filtfilt(b, a, signal)

b, a = sg.butter(3, [1/nyq, 4/nyq], btype='bandpass')
delta = sg.filtfilt(b, a, signal)


delta_range = np.where((1 < f) & (f < 4))[0]
delta_power = Sxx[delta_range, :]
delta_mean_power = np.mean(delta_power, axis=0)

theta_range = np.where((5 < f) & (f < 10))[0]
theta_power = Sxx[theta_range, :]
theta_mean_power = np.mean(theta_power, axis=0)

deltaplus_range = np.where((12 < f) & (f < 15))[0]
deltaplus_power = Sxx[deltaplus_range, :]
deltaplus_mean_power = np.mean(deltaplus_power, axis=0)

delta_theta_ratio = theta_mean_power/(delta_mean_power + deltaplus_mean_power)

windowLength = SampFreq/SampFreq*30
window = np.ones((int(windowLength),))/windowLength


delta_theta_ratio = sg.filtfilt(window, 1, delta_theta_ratio, axis=0)
# delta_theta_ratio = smth.gaussian_filter1d(delta_theta_ratio, sigma=25)

edges = np.linspace(0, 8, 300)

hist_states, binedges = np.histogram(delta_theta_ratio, bins=edges)

f1 = coher[0]
m = coher[1]

plt.clf()

plt.subplot(211)
plt.plot(t/3600, delta_theta_ratio)


plt.subplot(212)

# plt.plot(binedges[0:len(binedges)-1], hist_states)
plt.plot(delta_theta_ratio, coher, '.')
# plt.ylim(-2, 2)
# plt.pcolormesh(t/3600, f, Sxx, cmap='copper', vmax=30)
