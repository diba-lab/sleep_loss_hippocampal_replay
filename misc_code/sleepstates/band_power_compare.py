import numpy as np
from callfunc import processData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import pandas as pd
import seaborn as sns

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

group = []
# for sub, sess in enumerate(sessions[:3]):

#     sess.trange = np.array([])
#     t_start = sess.epochs.post[0] + 5 * 3600
#     df = sess.brainstates.states
#     df = df.loc[(df["state"] == 1) & (df["duration"] > 300)]
#     # df = df[:1]
#     df["condition"] = ["sd"] * len(df)

#     params = sess.brainstates.params

#     theta_delta = []
#     for i in range(len(df)):
#         start = df.start.iloc[i]
#         end = df.end.iloc[i]
#         val = params.loc[(params["time"] > start) & (params["time"] < end), "spindle"]
#         theta_delta.append(np.mean(val))
#         # break
#     df["ratio"] = theta_delta
#     group.append(df)

delta, ripple = [], []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    t_start = sess.epochs.post[0]
    df = sess.brainstates.states
    df = df.loc[(df["state"] == 1) & (df["duration"] > 300)]
    # df = df[:1]
    df["condition"] = ["nsd"] * len(df)

    params = sess.brainstates.params

    # theta_delta = []
    for i in range(len(df)):
        start = df.start.iloc[i]
        end = df.end.iloc[i]
        bins = np.linspace(start + 1, end - 60, 40)
        t1 = params.loc[(params["time"] > start) & (params["time"] < end), "time"]
        val1 = params.loc[(params["time"] > start) & (params["time"] < end), "delta"]
        val2 = params.loc[(params["time"] > start) & (params["time"] < end), "ripple"]

        delta_ind = np.interp(bins, t1, val1)
        ripple_ind = np.interp(bins, t1, val2)
        delta.append(delta_ind)
        ripple.append(ripple_ind)
        # break

    # df["ratio"] = theta_delta
    group.append(df)

delta = np.asarray(delta)
ripple = np.asarray(ripple)
# group = pd.concat(group, ignore_index=True)

plt.plot(np.mean(delta, axis=0))
plt.plot(np.mean(ripple, axis=0))

# ax = sns.boxplot(x="condition", y="ratio", data=group, palette="Set3")
# ax.set_ylim(-10, 2000)
# ax.set_ylabel("ripple amplitude")
# ax.set_xlabel("")
# ax.set_xticklabels(["nrem", "rem"])
# ax.set_title("peak theta/delta ratio during rem")
