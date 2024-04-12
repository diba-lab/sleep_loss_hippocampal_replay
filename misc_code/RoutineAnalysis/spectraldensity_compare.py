import numpy as np
import matplotlib.pyplot as plt
import os
from sleepDetect import SleepScore
import pandas as pd
from parsePath import name2path
import seaborn as sns
from scipy import stats
import scipy.fftpack as ffts
import scipy.ndimage as filtSig
from scipy.cluster import vq
from gwpy.timeseries import TimeSeries


sns.set(style="whitegrid")


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
freq = np.linspace(1, 500, 5000)

sws_dur = []
psd_sess_pre, psd_sess_post = [], []
for i, sess in enumerate(sessions):

    thetachan = np.load(str(sess.filePrefix) + "_BestThetaChan.npy", allow_pickle=True)
    # thetachan = TimeSeries(thetachan, sample_rate=1250)
    # thetachan = thetachan.whiten(4, 2)
    # thetachan = vq.whiten(thetachan)
    record_dur = len(thetachan) / 1250
    theta_t = np.linspace(0, record_dur, len(thetachan))
    deltastates = np.load(str(sess.filePrefix) + "_sws.npy", allow_pickle=True)
    epochs = np.load(str(sess.filePrefix) + "_epochs.npy", allow_pickle=True)
    pre = epochs.item().get("PRE")  # in seconds
    maze = epochs.item().get("MAZE")  # in seconds
    post = epochs.item().get("POST")  # in seconds

    states = deltastates.item().get("sws_epochs")
    states_dur = np.diff(states, axis=1)
    states = np.hstack((states, states_dur))

    dur_thresh = 420
    if i in [0, 2, 4]:
        states_pre = states[(states[:, 1] < pre[1]) & (states[:, 2] > dur_thresh), :]
        states_post = states[
            (states[:, 0] > post[0] + 5 * 3600) & (states[:, 2] > dur_thresh), :
        ]
        grp = "sd"

    else:
        states_pre = states[(states[:, 1] < pre[1]) & (states[:, 2] > dur_thresh), :]
        states_post = states[(states[:, 0] > post[0]) & (states[:, 2] > dur_thresh), :]
        grp = "nsd"

    psd_pre, psd_post = [], []
    # for st in states_pre:
    #     ind = np.where((theta_t > st[0]) & (theta_t < st[1]))
    #     theta_state = thetachan[ind]
    #     N = len(theta_state)
    #     # sample spacing
    #     T = 1.0 / 1250.0
    #     yf = ffts.fft(theta_state)
    #     xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

    #     psd_st = np.interp(freq, xf, 2.0 / N * np.abs(yf[: N // 2]))
    #     psd_pre.append(psd_st)

    # psd_pre = np.asarray(psd_pre)
    # psd_pre_mean = np.mean(psd_pre, axis=0)
    # psd_sess_pre.append(psd_pre_mean)

    for st in states_post:
        ind = np.where((theta_t > st[0]) & (theta_t < st[1]))
        theta_state = thetachan[ind]
        N = len(theta_state)
        # sample spacing
        T = 1.0 / 1250.0
        yf = ffts.fft(theta_state)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

        psd_st = np.interp(freq, xf, 2.0 / N * np.abs(yf[: N // 2]))
        psd_post.append(psd_st)

    psd_post = np.asarray(psd_post)
    psd_post_mean = np.mean(psd_post, axis=0)
    psd_sess_post.append([psd_post_mean, grp])


# psd_data = pd.DataFrame({"freq": freq, "mean_post": psd_sess_post})

# ax = sns.lineplot(x="freq", y="mean_psd", data=psd_data)
plt.clf()
for ft_post in psd_sess_post:
    ft_post[0] = filtSig.gaussian_filter1d(ft_post[0], 2)

    if ft_post[1] == "sd":
        plt.plot(freq, ft_post[0], color="red")
    else:
        plt.plot(freq, ft_post[0], color="green")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude (a.u.)")
# plt.legend([x.sessionName for x in sessions])
plt.legend(["sd", "nsd"])
plt.savefig("test_1.png", dpi=150)
plt.title(
    "Comparison of frequencies in sws epochs (>7 minutes) between recovery and normal sleep "
)

