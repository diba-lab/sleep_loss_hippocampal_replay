import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from getPosition import ExtractPosition as getPos
import matplotlib


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]
nSessions = len(basePath)

posSession = [getPos(basePath[i]) for i in range(nSessions)]
for i in range(nSessions):

    posSession[i].getMazeFrames()

# plt.plot(posSession[i].frames, posSession[i].posZ)
# RatJDay2 = getPos(basePath)
# velocity = RatNDay2.Speed()
# plt.clf()
# plt.plot(velocity)

# a = [1, 2, 3, 4]
# b = [5, 6, 7, 8]
# plt.clf()
# plt.plot(a, b)
# pts = plt.ginput(2, timeout=-1)

# basePath = "/data/Clustering/SleepDeprivation/RatJ/Day2/"
# subname = "RatJ_Day2_2019-06-02_03-59-19"
# os.path.exists(basePath + subname + "/og_files")
