#%%====== Import statements==========
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SpectralAnalysis import bestRippleChannel, bestThetaChannel, lfpSpectrogram
from lfpDetect import swr
import os
from datetime import datetime as dt

import matplotlib.style
import matplotlib as mpl

sns.set(style="whitegrid")
# mpl.style.use("default")

# %load_ext autoreload
# %autoreload 2


#  ========== Ripple Detection ============


class RippleDetect:
    sRate = 1250

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".eeg"):
                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])

    def findRipples(self):
        epoch_time = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)
        recording_dur = epoch_time.item().get("POST")[1]  # in seconds
        pre = epoch_time.item().get("PRE")  # in seconds
        print(pre[1])
        maze = epoch_time.item().get("MAZE")  # in seconds
        post = epoch_time.item().get("POST")  # in seconds
        self.basics = np.load(self.filePrefix + "_basics.npy", allow_pickle=True)
        self.nChans = self.basics.item().get("nChans")

        if not os.path.exists(self.filePrefix + "_BestRippleChans.npy"):

            badChannels = np.load(self.filePrefix + "_badChans.npy")
            self.bestRippleChannels = bestRippleChannel(
                self.basePath,
                sampleRate=self.sRate,
                nChans=self.nChans,
                badChannels=badChannels,
                saveRippleChan=1,
            )

            swr(self.basePath, sRate=self.sRate, PlotRippleStat=0, savefile=1)

        self.ripples = np.load(self.filePrefix + "_ripples.npy", allow_pickle=True)

        self.ripplesTime = self.ripples.item().get("timestamps")
        self.rippleStart = self.ripplesTime[:, 0] / 1250
        self.histRipple, self.edges = np.histogram(self.rippleStart, bins=20)
        self.edges = self.edges / (1250 * 3600)

        bin_epoch = [
            pre[0],
            pre[1],
            post[0],
            post[0] + 5 * 3600,
            post[0] + 5 * 3600,
            post[1],
        ]
        bin_dur = np.diff(bin_epoch)
        self.epochcount, _ = np.histogram(self.rippleStart, bins=bin_epoch)
        self.ripprate = self.epochcount[::2] / bin_dur[::2]
        # print(bin_dur / 3600)

    # def lfpSpect(self):
    #     self.Pxx, self.freq, self.time, self.sampleData = lfpSpectrogram(
    #         self.basePath, self.sRate, nChans=self.nChans, loadfrom=1
    #     )
    #     f_req_ind = np.where(self.freq < 50)[0]

    #     self.f_req = self.freq[f_req_ind]
    #     self.Pxx_req = self.Pxx[f_req_ind, :]
    #     self.Pxx_req = np.flipud(self.Pxx_req)
    #     self.time = self.time / 3600

    def sessionInfo(self):
        self.Date = self.ripples["DetectionParams"]


basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

Ripple_inst = [RippleDetect(x) for x in basePath]


for i, sess in enumerate(Ripple_inst):
    sess.findRipples()


plt.clf()
ripp_recovery, ripp_nsd = [], []
for i, sess in enumerate(Ripple_inst[::2]):
    ripp_recovery.append(sess.ripprate[-1])
for i, sess in enumerate(Ripple_inst[1::2]):
    ripp_nsd.append(sess.ripprate[-2])

ripp_rate = [ripp_nsd, ripp_recovery]
grp_label = ["NSD", "Recovery sleep"]

ax = sns.barplot(data=ripp_rate)
ax.set_xticklabels(grp_label)
ax.set_ylabel("Ripple rate (Hz)")
ax.set_title("Comparing ripple rate")
# plt.legend([x.subname[:9] for x in Ripple_inst])
# plt.xlim([-2, 4])
# plt.title("Ripple rate")
# plt.ylabel("Ripple rate (Hz)")
# # plt.xlabel("Ripple rate (Hz)")
# plt.xticks([0, 1, 2], ["PRE", "SD", "POST"])

