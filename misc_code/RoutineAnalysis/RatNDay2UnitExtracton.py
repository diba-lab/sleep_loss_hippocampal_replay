import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# recording_file = np.memmap(
#     "/data/Clustering/SleepDeprivation/RatN/Day2/RatN-Day2-2019-10-11_03-58-54.dat",
#     dtype="int16",
#     mode="r",
# )


class ExtractSpikes:

    nChans = 16
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # in seconds

    def __init__(self, basePath):
        # self.sessionnName = os.path.basename(os.path.normpath(basePath))
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".eeg"):
                print(file)
                self.subname = file[:-4]
                print(os.path.join(basePath, file))
                self.filePrefix = os.path.join(basePath, file[:-4])

    def CollectSpikes(self):
        self.spkAll = []
        # firing_rate = []

        for i in range(1, 9):
            fileName = self.basePath + "Shank" + str(i) + "/"
            spike_times = np.load(fileName + "spike_times.npy")
            cluster_labels = pd.read_csv(fileName + "cluster_q.tsv", sep="\t")
            clusterInfo = pd.read_csv(fileName + "cluster_info.tsv", sep="\t")
            clusterInfo.head(5)
            goodCellsID = cluster_labels.cluster_id[cluster_labels["q"] == 1].tolist()
            # goodcells_firingRate = clusterInfo.firing_rate[
            #     clusterInfo["group"] == "good"
            # ].tolist()
            cluID = np.load(fileName + "spike_clusters.npy")

            # firing_rate.append(goodcells_firingRate)
            for i in range(len(goodCellsID)):
                clu_spike_location = spike_times[np.where(cluID == goodCellsID[i])[0]]
                self.spkAll.append(clu_spike_location / self.sRate)
        np.save(self.basePath + self.subname + "_spikes.npy", self.spkAll)

    def partialCorrelation(self, X, Y, Z):
        corrXY = [pd.Series.corr(pd.Series(X), pd.Series(Y[i])) for i in range(len(Y))]
        corrYZ = [pd.Series.corr(pd.Series(Z), pd.Series(Y[i])) for i in range(len(Y))]
        corrXZ = pd.Series.corr(pd.Series(X), pd.Series(Z))

        parCorr = [
            (corrXY[m] - corrXZ * corrYZ[m])
            / (np.sqrt(1 - corrXZ ** 2) * (np.sqrt(1 - corrYZ[m] ** 2)))
            for m in range(len(corrYZ))
        ]

        revCorr = [
            (corrXZ - corrXY[m] * corrYZ[m])
            / ((np.sqrt(1 - corrXY[m] ** 2)) * (np.sqrt(1 - corrYZ[m] ** 2)))
            for m in range(len(corrYZ))
        ]

        return parCorr, revCorr

    def ExpVAr(self):

        epoch_time = np.load(self.filePrefix + "_epochs.npy", allow_pickle=True)
        recording_dur = epoch_time.item().get("POST")[1]  # in seconds
        pre = epoch_time.item().get("PRE")  # in seconds
        maze = epoch_time.item().get("MAZE")  # in seconds
        post = epoch_time.item().get("POST")  # in seconds

        self.nUnits = len(self.spkAll)
        windowSize = self.timeWindow

        # Calculating firing rate
        f_rate = [len(x) / recording_dur for x in self.spkAll]
        pyr_id = [x for x in range(len(f_rate)) if f_rate[x] < 10]
        spkAll = [self.spkAll[x] for x in pyr_id]

        pre_bin = np.arange(pre[0], pre[1], 0.250)
        maze_bin = np.arange(maze[0], maze[1], 0.250)
        post_bin = np.arange(post[0], post[1], 0.250)

        pre_spikecount = np.array([np.histogram(x, bins=pre_bin)[0] for x in spkAll])
        maze_spikecount = np.array([np.histogram(x, bins=maze_bin)[0] for x in spkAll])
        post_spikecount = np.array([np.histogram(x, bins=post_bin)[0] for x in spkAll])
        post_spikecount = [
            post_spikecount[:, i : i + windowSize]
            for i in range(0, 10 * windowSize, windowSize)
        ]

        pre_corr = np.corrcoef(pre_spikecount)
        maze_corr = np.corrcoef(maze_spikecount)
        post_corr = [
            np.corrcoef(post_spikecount[x]) for x in range(len(post_spikecount))
        ]

        self.check1 = post_corr
        # selecting only lower diagonal of matrix
        pre_corr = pre_corr[np.tril_indices(pre_corr.shape[0], k=-1)]
        maze_corr = maze_corr[np.tril_indices(maze_corr.shape[0], k=-1)]
        post_corr = [
            post_corr[x][np.tril_indices(post_corr[x].shape[0], k=-1)]
            for x in range(len(post_spikecount))
        ]

        self.check2 = post_corr
        parcorr_maze_vs_post, revcorr = self.partialCorrelation(
            maze_corr, post_corr, pre_corr
        )

        self.ev_maze_vs_post = [x ** 2 for x in parcorr_maze_vs_post]
        self.rev = [x ** 2 for x in revcorr]

        np.save(self.basePath + self.subname + "_EV.npy", self.ev_maze_vs_post)
        np.save(self.basePath + self.subname + "_REV.npy", self.rev)

        # return ev_maze_vs_post

    # def FiringRate(self):
    #     spike_corr_diff = [
    #         np.histogram(
    #             self.spkAll[x], bins=np.arange(0, 16 * windowSize, windowSize)
    #         )[0]
    #         for x in range(0, nUnits)
    #     ]

    #     f_rate_hist = np.histogram(f_rate, bins=np.arange(0, 30, 0.1))

    # def sessionInfo(self):
    #     self.Date = self.ripples["DetectionParams"]


folderPath = "/data/Clustering/SleepDeprivation/RatN/Day1/"

RatNDay1 = ExtractSpikes(folderPath)
RatNDay1.CollectSpikes()
RatNDay1.ExpVAr()


# folderPath = "/data/Clustering/SleepDeprivation/RatN/Day2/"

# RatNDay2 = ExtractSpikes(folderPath)
# RatNDay2.CollectSpikes()
# RatNDay2.ExpVAr()

# plt.clf()
# plt.plot(RatNDay1.ev_maze_vs_post)
# plt.plot(RatNDay2.ev_maze_vs_post, "r")

