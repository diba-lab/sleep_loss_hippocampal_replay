import numpy as np
import matplotlib.pyplot as plt

# TODO delete noisy artifacts

filename = "/data/Clustering/SleepDeprivation/RatK/Day1/2019-08-06_03-44-01/experiment1/recording1/continuous/Rhythm_FPGA-100.0/Shank4.dat"


nChans = 16
SampFreq = 30000
duration = 200 * 60
b1 = np.memmap(filename, dtype="int16", mode="r", shape=(nChans * SampFreq * duration))

duration2 = 207 * 60
b2 = np.memmap(
    filename, dtype="int16", mode="r", offset=2 * nChans * SampFreq * duration2
)


c = np.memmap("Shank4_NoNoise.dat", dtype="int16", mode="w+", shape=(len(b1) + len(b2)))

del c
d = np.memmap("Shank4_NoNoise.dat", dtype="int16", mode="r+", shape=(len(b1) + len(b2)))
d[: len(b1)] = b1
d[len(b1) : len(b1) + len(b2)] = b2

