import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from callfunc import processData

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]


sess = processData(basePath[0])

# badChans = [101, 105, 114, 126, 127] + list(range(128, 134))

# for sub, sess in enumerate(sessions):

#     # sess.recinfo.makerecinfo(badchans=badChans)
# sess.trange = np.array([])
# sess.makePrmPrb.makePrbCircus(probetype="diagbio")
# sess.recinfo.makerecinfo()
sess.artifact.art_thresh = 9
zsc_signal = sess.artifact.usingZscore()

plt.plot(zsc_signal)
