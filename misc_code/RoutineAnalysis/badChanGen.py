import numpy as np
import os

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    "/data/Clustering/SleepDeprivation/RatJ/Day4/"
]


chanlist = [
    # [0, 2, 6, 5] + list(range(64, 75)),
    # [0, 2, 6, 5] + list(range(64, 67)),
    # list(range(64, 134)),
    # [101, 105, 126, 127] + list(range(128, 134)),
    # [13, 14, 15, 63] + list(range(128, 134)),
    # [13, 14, 15, 63] + list(range(128, 134)),
    [0, 2, 6, 4, 5, 47]
    + list(range(64, 67))
]

for i, session in enumerate(basePath):
    for file in os.listdir(session):
        if file.endswith(".eeg"):
            subname = file[:-4]
            print(subname)

    np.save(session + subname + "_badChans.npy", chanlist[i])
    chans = np.load(session + subname + "_badChans.npy")
    print(chans)
