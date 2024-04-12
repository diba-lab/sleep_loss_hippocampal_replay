import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

folderPath = "/data/Clustering/SleepDeprivation/RatN/Day1/"
fileName = folderPath + "RatN_Day1_2019-10-09_03-52-32.eeg"

fileOGDat = folderPath + "RatN_Day1_2019-10-09_03-52-32.dat"

nChansEEG = 134
SampFreq = 1250


Data = np.memmap(fileName, dtype="int16", mode="r")
# Data2 = np.memmap(fileOGDat, dtype="int16", mode="r")


Data1 = np.memmap.reshape(Data, (int(len(Data) / nChansEEG), nChansEEG))

chanData = Data1[:, 17]

zsc = np.abs(stat.zscore(chanData))

artifact_binary = np.where(zsc > 1.5, 1, 0)
# for i in range(2, 9):

#     filename = (
#         "/data/Clustering/SleepDeprivation/RatN/Day1/Shank"
#         + str(i)
#         + "/Shank"
#         + str(i)
#         + ".dat"
#     )

#     nChans = 16
#     SampFreq = 30000
#     duration = 191 * 60  # in seconds
#     b1 = np.memmap(
#         filename, dtype="int16", mode="r", shape=(nChans * SampFreq * duration)
#     )

#     duration2 = 192 * 60  # seconds
#     b2 = np.memmap(
#         filename, dtype="int16", mode="r", offset=2 * nChans * SampFreq * duration2
#     )

#     DestFile = (
#         "/data/Clustering/SleepDeprivation/RatN/Day1/Shank"
#         + str(i)
#         + "/RatNDay1Shank"
#         + str(i)
#         + ".dat"
#     )
#     c = np.memmap(DestFile, dtype="int16", mode="w+", shape=(len(b1) + len(b2)))

#     del c
#     d = np.memmap(DestFile, dtype="int16", mode="r+", shape=(len(b1) + len(b2)))
#     d[: len(b1)] = b1
#     d[len(b1) : len(b1) + len(b2)] = b2

