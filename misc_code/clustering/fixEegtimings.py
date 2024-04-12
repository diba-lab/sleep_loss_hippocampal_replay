import numpy as np
import matplotlib.pyplot as plt


filename = (
    "/data/Clustering/SleepDeprivation/RatN/Day1/RatN_Day1_2019-10-09_03-52-32_og.eeg"
)

Destfile = (
    "/data/Clustering/SleepDeprivation/RatN/Day1/RatN_Day1_2019-10-09_03-52-32.eeg"
)

nChans = 134
SampFreq = 1250
duration1 = 191  # from this time in seconds
duration2 = 60  # duration of chunk

# read required chunk from the source file
b1 = np.memmap(filename, dtype="int16", mode="r")


# allocates space for the file
c = np.memmap(Destfile, dtype="int16", mode="w+", shape=(len(b1) - (60 * 1250 * 134)))
c[: 191 * 60 * 1250 * 134] = b1[: 191 * 60 * 1250 * 134]
c[191 * 60 * 1250 * 134 :] = b1[192 * 60 * 1250 * 134 :]
# del c

# writes the data to that space
d = np.memmap(Destfile, dtype="int16", mode="r+", shape=(len(b1)))
