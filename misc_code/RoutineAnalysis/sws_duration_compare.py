import numpy as np
import matplotlib.pyplot as plt
import os
from sleepDetect import SleepScore
import pandas as pd
from parsePath import name2path
import seaborn as sns
from bokeh.io import show, output_file
from bokeh.plotting import figure
from scipy import stats
import altair as alt

sns.set(style="whitegrid")

figurepath = "/data/DataGen/figuresGen/SleepDeprivation/"
output_file(figurepath + "swsduration.html")

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]

sessions = [name2path(_) for _ in basePath]

# for i, sess in enumerate(sessions):
#     sess.deltaStates()


sws_dur = []
for i, sess in enumerate(sessions):

    deltastates = np.load(str(sess.filePrefix) + "_sws.npy", allow_pickle=True)
    epochs = np.load(str(sess.filePrefix) + "_epochs.npy", allow_pickle=True)
    pre = epochs.item().get("PRE")  # in seconds
    maze = epochs.item().get("MAZE")  # in seconds
    post = epochs.item().get("POST")  # in seconds

    states = deltastates.item().get("sws_epochs")
    states = deltastates.item().get("sws_epochs")
    states = deltastates.item().get("sws_epochs")
    states_dur = np.diff(states, axis=1)
    states = np.hstack((states, states_dur))

    if i in [0, 2, 4]:
        states = states[(states[:, 0] > post[0] + 5 * 3600) & (states[:, 2] > 100), :]
        grp = "sd"

    else:
        states = states[(states[:, 0] > post[0]) & (states[:, 2] > 100), :]
        grp = "nsd"

    mean_sws = np.mean(states[:, 2])

    sws_dur.append([mean_sws, grp])


data = pd.DataFrame(sws_dur, columns=["swsdur", "group"])

# colormap = {"Day1": "red", "Day2": "green"}
# colors = [colormap[x.split("_")[1]] for x in data["name"]]

plt.clf()
ax = sns.barplot(x="group", y="swsdur", data=data, ci="sd", palette=["red", "green"])
ax.set(xlabel="", ylabel="SWS duration (s)")


# source = pd.DataFrame(
#     {
#         "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
#         "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
#     }
# )
# alt.Chart(data).mark_bar().encode(x="group", y="swsdur")

# alt.Chart(source).mark_bar().encode(x="a", y="b")

