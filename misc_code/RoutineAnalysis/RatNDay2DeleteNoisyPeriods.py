import numpy as np
import matplotlib.pyplot as plt

# TODO delete noisy artifacts

filename = (
    "/data/Clustering/SleepDeprivation/RatN/Day2/RatN-Day2-2019-10-11_03-58-54.dat"
)

DestFolder = "/data/Clustering/SleepDeprivation/RatN/Day2/RatN-Day2-2019-10-11_03-58-54_NoNoise.dat"

nChans = 134
SampFreq = 30000
duration = 168 * 60  # in seconds
b1 = np.memmap(filename, dtype="int16", mode="r", shape=(nChans * SampFreq * duration))

duration2 = 172 * 60
b2 = np.memmap(
    filename, dtype="int16", mode="r", offset=2 * nChans * SampFreq * duration2
)


c = np.memmap(DestFolder, dtype="int16", mode="w+", shape=(len(b1) + len(b2)))

del c
d = np.memmap(DestFolder, dtype="int16", mode="r+", shape=(len(b1) + len(b2)))
d[: len(b1)] = b1
d[len(b1) : len(b1) + len(b2)] = b2

