#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os
from datetime import datetime as dt
import matplotlib.style
import matplotlib as mpl
import scipy.signal as sg

mpl.style.use("default")
# %load_ext autoreload
# %autoreload 2


#%% ======== Theta detection testuing==============
folderPath = [
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-05-31_03-55-36/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-08_04-00-00/",
]

badChannels = [np.arange(65, 135), [1, 3, 7, 6, 65, 66, 67]]

nChans_all = [75, 67, 134, 134]

plt.clf()
for i in [0, 1, 2, 3]:
    basePath = folderPath[i]
    sampleRate = 1250
    numChans = nChans_all[i]
    subname = os.path.basename(os.path.normpath(basePath))

    # ripples = swr(basePath, sRate=sampleRate, PlotRippleStat=0)
    Pxx, f, t, samp_data = lfpSpectrogram(
        basePath, sRate=sampleRate, loadfrom=1, nChans=numChans
    )
    f_req_ind = np.where(f < 50)[0]

    f_req = f[f_req_ind]
    Pxx_req = Pxx[f_req_ind, :]
    Pxx_req = np.flipud(Pxx_req)
    plt.subplot(4, 1, i + 1)
    plt.imshow(
        np.log10(Pxx_req),
        extent=[0, t[-1] / 3600, f_req[0], f_req[-1]],
        aspect="auto",
        vmax=7,
        vmin=2,
        cmap="copper",
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (h)")


# %% ========== Ripple Detection ============


class RippleDetect:
    nChans = 134
    sRate = 1250
    badChannels = np.arange(65, 134)

    def __init__(self, basePath):
        self.sessionnName = os.path.basename(os.path.normpath(basePath))
        self.basePath = basePath

    def findRipples(self):
        if not os.path.exists(
            self.basePath + self.sessionnName + "_BestRippleChans.npy"
        ):
            self.bestRippleChannels = bestRippleChannel(
                self.basePath,
                sampleRate=self.sRate,
                nChans=self.nChans,
                badChannels=self.badChannels,
                saveRippleChan=1,
            )
        self.Starttime = dt.strptime(self.sessionnName[-19:], "%Y-%m-%d_%H-%M-%S")
        self.ripples = swr(self.basePath, sRate=self.sRate, PlotRippleStat=1)
        self.ripplesTime = self.ripples["timestamps"]
        self.rippleStart = self.ripplesTime[:, 0]
        self.histRipple, self.edges = np.histogram(self.rippleStart, bins=20)

    def lfpSpect(self):
        self.spect, self.freq, self.time, self.sampleData = lfpSpectrogram(
            self.basePath, self.sRate, nChans=self.nChans, loadfrom=1
        )
        self.time = self.time / 3600
        self.spect = np.flip(self.spect)

    def sessionInfo(self):
        self.Date = self.ripples["DetectionParams"]


folderPath = [
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-05-31_03-55-36/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-08_04-00-00/",
]


RatJ_SD = RippleDetect(folderPath[0])
RatJ_NoSD = RippleDetect(folderPath[1])
RatK_SD = RippleDetect(folderPath[2])
RatK_NoSD = RippleDetect(folderPath[3])

RatJ_NoSD.badChannels = [1, 3, 7, 6, 65, 66, 67]
RatJ_NoSD.nChans = 67
# RatJ_NoSD.findRipples()
RatJ_NoSD.lfpSpect()

# RatJ_SD.badChannels = [1, 3, 7] + list(range(65, 76))
# RatJ_SD.nChans = 75
# # RatJ_SD.findRipples()
# RatJ_SD.lfpSpect()

# # RatK_SD.findRipples()
# RatK_SD.lfpSpect()

# RatK_NoSD.badChannels = [102, 106, 127, 128]
# # RatK_NoSD.findRipples()
# RatK_NoSD.lfpSpect()

sessions = ["RatJ_SD", "RatJ_NoSD", "RatK_SD", "RatK_NoSD"]
spect_sessions = [RatJ_NoSD.spect]

plt.clf()
for i in range(1):
    plt.subplot(4, 1, i + 1)
    plt.imshow(
        RatJ_NoSD.spect[10:200, :],
        cmap="YlGn",
        vmax=0.01,
        extent=[
            np.min(RatJ_NoSD.time),
            np.max(RatJ_NoSD.time),
            np.min(RatJ_NoSD.freq),
            np.max(RatJ_NoSD.freq),
        ],
        aspect="auto",
    )


# plt.plot(RatJ_NoSD.histRipple)
# plt.plot(RatK_SD.histRipple, "k")
# plt.plot(RatK_NoSD.histRipple, "g")


# %%
