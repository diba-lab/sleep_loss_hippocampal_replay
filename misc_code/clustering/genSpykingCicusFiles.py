import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from callfunc import processData


# TODO thoughts on using data class for loading data into function

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/"
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/"
]


sessions = [processData(_) for _ in basePath]

for sub, sess in enumerate(sessions):

    # sess.recinfo.makerecinfo()
    sess.makePrmPrb.makePrbCircus("buzsaki")
    # sig_zsc = sess.artifact.usingZscore()


# plt.plot(sig_zsc)
