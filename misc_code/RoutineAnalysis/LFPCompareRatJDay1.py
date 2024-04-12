import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

fileName = "/data/Clustering/SleepDeprivation/RatJ/Day2/continuous.eeg"


nChans = 8
SampFreq = 1250


Data = np.memmap(fileName, dtype="int16", mode="r")

Data1 = np.memmap.reshape(Data, (int(len(Data) / 67), 67))

chanData = Data1[:, 17]

zsc = np.abs(stat.zscore(chanData))
plt.clf()
plt.plot(zsc)
# artifact_binary = np.where(zsc > 3, 0, 1)
# artifact_binary = np.concatenate(([0], artifact_binary, [0]))
# artifact_diff = np.diff(artifact_binary)


# artifact_start = np.where(artifact_diff == 1)[0] / 1250
# artifact_end = np.where(artifact_diff == -1)[0] / 1250


# fileName = (
#     "/data/Clustering/SleepDeprivation/RatJ/Day1/Shank2/RatJDay1_Shank2_Denoised.eeg"
# )


# nChans = 8
# SampFreq = 1250


# Data = np.memmap(fileName, dtype="int16", mode="r")

# Data1 = np.memmap.reshape(Data, (int(len(Data) / 8), 8))

# chanData = Data1[:, 2]

# zsc_denoised = stat.zscore(chanData)

# DenoisedData = np.where(zsc > 3, 0, 1)
# artifact_binary = np.concatenate(([0], artifact_binary, [0]))
# artifact_diff = np.diff(artifact_binary)


# artifact_start = np.where(artifact_diff == 1)[0] / 1250
# artifact_end = np.where(artifact_diff == -1)[0] / 1250
