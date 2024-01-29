import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sg
from SpectralAnalysis import bestThetaChannel
from sklearn import preprocessing
from scipy.cluster import vq
import matplotlib as mpl


mpl.rc("axes", linewidth=1.5)
mpl.rc("font", size=12)

# import multiprocessing as mp


class SDspect(object):

    sampFreq = 1250

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".eeg"):
                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])
                # print(os.path.join(basePath, file))

    def SpectCalculate(self):
        epoch_time = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)
        recording_dur = epoch_time.item().get("POST")[1]  # in seconds
        pre = epoch_time.item().get("PRE")  # in seconds
        maze = epoch_time.item().get("MAZE")  # in seconds
        post = epoch_time.item().get("POST")  # in seconds
        self.basics = np.load(self.filePrefix + "_basics.npy", allow_pickle=True)
        self.nChans = self.basics.item().get("nChans")

        if not os.path.exists(self.basePath + self.subname + "_BestThetaChan.npy"):

            badChannels = np.load(self.filePrefix + "_badChans.npy")

            self.ThetaChan = bestThetaChannel(
                self.basePath,
                sampleRate=self.sampFreq,
                nChans=self.nChans,
                badChannels=badChannels,
                saveThetaChan=1,
            )

        lfpData = np.load(self.filePrefix + "_BestThetaChan.npy", allow_pickle=True)

        start_time = int(post[0]) * self.sampFreq
        end_time = int(post[0]) * self.sampFreq + 5 * 3600 * self.sampFreq

        lfpSD = lfpData[start_time:end_time]
        # lfpSD = preprocessing.scale(lfpSD)
        lfpSD = vq.whiten(lfpSD)
        # plt.plot(lfpSD[: 1250 * 3])
        f, t, sxx = sg.spectrogram(
            lfpSD,
            fs=self.sampFreq,
            nperseg=5 * self.sampFreq,
            noverlap=2.5 * self.sampFreq,
        )

        freq_ind = np.where((f > 1) & (f < 50))[0]
        self.freq = f[freq_ind]
        self.req_sxx_mean = np.mean(sxx[freq_ind, :], axis=1)
        self.req_sxx_std = np.std(sxx[freq_ind, :], axis=1)


folderPath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]
# nChans_sessions = [75, 67, 134, 134, 134, 134]

nSessions = len(folderPath)
sleepDep_inst = [SDspect(folderPath[i]) for i in range(nSessions)]
# sessRun = range(4, nSessions)
sessRun = [0, 2, 4]

color = ["#cb7c7c", "#0a9eb8", "#cb7c7c", "#0a9eb8", "#cb7c7c", "#0a9eb8"]

for i in sessRun:

    sleepDep_inst[i].SpectCalculate()
    print(sleepDep_inst[i].nChans)


fig = plt.figure(1)
plt.clf()
k = 1
for i in sessRun:
    # plt.subplot(4, 1, i + 1)
    # ax1 = fig.add_subplot(len(sessRun), 1, k)

    plt.plot(sleepDep_inst[i].freq, sleepDep_inst[i].req_sxx_mean)
    # ax1.plot(
    #     sleepDep_inst[i].freq,
    #     sleepDep_inst[i].req_sxx_mean + sleepDep_inst[i].req_sxx_std,
    # )
    plt.yscale("log")
    k += 1

plt.legend([sleepDep_inst[x].subname[:9] for x in sessRun])
plt.ylabel("Power (A.U.)")
plt.xlabel("Frequency (Hz)")


# plt.axis("off")

