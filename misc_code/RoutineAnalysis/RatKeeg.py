#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os
from datetime import datetime as dt

import matplotlib.style
import matplotlib as mpl

mpl.style.use("default")

# %load_ext autoreload
# %autoreload 2


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
        self.ripples = swr(self.basePath, sRate=self.sRate, PlotRippleStat=0)
        self.ripplesTime = self.ripples["timestamps"]
        self.rippleStart = self.ripplesTime[:, 0]
        self.histRipple, self.edges = np.histogram(self.rippleStart, bins=20)
        self.edges = self.edges / (1250 * 3600)

    def lfpSpect(self):
        self.Pxx, self.freq, self.time, self.sampleData = lfpSpectrogram(
            self.basePath, self.sRate, nChans=self.nChans, loadfrom=1
        )
        f_req_ind = np.where(self.freq < 50)[0]

        self.f_req = self.freq[f_req_ind]
        self.Pxx_req = self.Pxx[f_req_ind, :]
        self.Pxx_req = np.flipud(self.Pxx_req)
        self.time = self.time / 3600

    def sessionInfo(self):
        self.Date = self.ripples["DetectionParams"]


folderPath = [
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-05-31_03-55-36/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatJ/RatJ_2019-06-02_03-59-19/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-06_03-44-01/",
    "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatK/RatK_2019-08-08_04-00-00/",
    # "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatN/RatN_2019-08-08_04-00-00/",
    # "/home/bapung/Documents/ClusteringHub/EEGAnlaysis/RatN/RatN_2019-08-08_04-00-00/",
]


badChannels_all = [
    [1, 3, 7] + list(range(65, 76)),
    [1, 3, 7, 6, 65, 66, 67],
    np.arange(65, 135),
    [102, 106, 127, 128],
]

nChans_all = [75, 67, 134, 134]

Ripple_inst = [RippleDetect(folderPath[i]) for i in range(4)]

for i in range(4):
    Ripple_inst[i].badChannels = badChannels_all[i]
    Ripple_inst[i].nChans = nChans_all[i]
    Ripple_inst[i].findRipples()
    Ripple_inst[i].lfpSpect()


fig = plt.figure(1)
for i in range(4):
    # plt.subplot(4, 1, i + 1)
    ax1 = fig.add_subplot(4, 1, i + 1)

    ax1.imshow(
        np.log(Ripple_inst[i].Pxx_req),
        cmap="OrRd",
        aspect="auto",
        vmax=12,
        vmin=4,
        extent=[
            Ripple_inst[i].time[0],
            Ripple_inst[i].time[-1],
            Ripple_inst[i].f_req[0],
            Ripple_inst[i].f_req[-1],
        ],
    )
    ax2 = ax1.twinx()
    ax2.plot(Ripple_inst[i].edges[:-1], Ripple_inst[i].histRipple, "k")
    plt.xlim([0, Ripple_inst[i].time[-1]])
    plt.title(Ripple_inst[i].sessionnName)

