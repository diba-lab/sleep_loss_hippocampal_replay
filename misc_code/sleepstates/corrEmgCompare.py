import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
import scipy.ndimage as filtSig
import time

from joblib import Parallel, delayed

from callfunc import processData


def getemg(eegdata, channels):
    highfreq = 600
    lowfreq = 300
    sRate = 1250
    nyq = 0.5 * sRate

    window = 1 * sRate
    overlap = 0.2 * sRate

    b, a = sg.butter(3, [lowfreq / nyq, highfreq / nyq], btype="bandpass")
    # print(b)

    # windowing signal
    frames = np.arange(0, len(eegdata) - window, window - overlap)

    tic = time.time()

    # results = []
    # for start in frames:
    def corrchan(start):
        start_frame = int(start)
        end_frame = start_frame + window
        lfp_req = eegdata[start_frame:end_frame, channels]
        yf = sg.filtfilt(b, a, lfp_req, axis=0)
        temp = yf.T
        # results.append(np.corrcoef(temp))
        return np.corrcoef(temp)

    # emg_lfp = [[] for _ in range(len(frames))]
    results = Parallel(n_jobs=8, require="sharedmem")(
        delayed(corrchan)(start) for start in frames
    )

    toc = time.time()

    print(toc - tic)

    return results


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sess = processData(basePath[0])


sRate = sess.recinfo.lfpSrate
nChans = sess.recinfo.nChans
nShanks = sess.recinfo.nShanks
channels = sess.recinfo.channels
changroup = sess.recinfo.channelgroups
# print(changroup)
badchans = sess.recinfo.badchans

if len(badchans) > 30:

    changroup = changroup[:4]
else:
    changroup = changroup[:nShanks]

chan_top = [
    np.setdiff1d(channels, badchans, assume_unique=True)[0] for channels in changroup
]
chan_middle = [
    np.setdiff1d(channels, badchans, assume_unique=True)[8] for channels in changroup
]
chan_bottom = [
    np.setdiff1d(channels, badchans, assume_unique=True)[-1] for channels in changroup
]

chan_map_select = np.setdiff1d(channels, badchans, assume_unique=True)

eegdata = np.memmap(sess.sessinfo.recfiles.eegfile, dtype="int16", mode="r")
eegdata = np.memmap.reshape(eegdata, (int(len(eegdata) / nChans), nChans))

corr_frame = getemg(eegdata, chan_map_select)

chan2emg = [
    [chan_top[0], chan_top[-1]],
    [chan_top[0], chan_bottom[-1]],
    chan_top + chan_bottom,
    chan_top + chan_middle + chan_bottom,
    list(chan_map_select),
]

emg = [[] for _ in range(len(chan2emg))]
for i, chn in enumerate(chan2emg):

    ind = [np.where(chan_map_select == _)[0][0] for _ in chn]
    print(ind)
    ltriang = np.tril_indices(len(ind), k=-1)
    ixgrid = np.ix_(ind, ind)
    emg[i] = [np.mean(frame[ixgrid][ltriang]) for frame in corr_frame]

emg_smth = [filtSig.gaussian_filter1d(_, 10) for _ in emg]
df = pd.DataFrame(
    {
        "top": emg_smth[0],
        "top_bottom": emg_smth[1],
        "sup_deep": emg_smth[2],
        "sup_middle_deep": emg_smth[3],
        "all": emg_smth[4],
    }
)

chanprobe = sess.recinfo.channels[:128]
coords = sess.recinfo.probemap()
x = np.asarray(coords[0])
y = np.asarray(coords[1])

fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(5, 6, figure=fig)
fig.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, y, "k.", markersize=3)
chan_req = [np.where(chanprobe == _)[0][0] for _ in chan2emg[0]]
ax1.plot(x[chan_req], y[chan_req], "r.")
ax1.axis("off")


ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x, y, "k.", markersize=3)
chan_req = [np.where(chanprobe == _)[0][0] for _ in chan2emg[1]]
ax2.plot(x[chan_req], y[chan_req], "r.")
ax2.axis("off")


ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(x, y, "k.", markersize=3)
chan_req = [np.where(chanprobe == _)[0][0] for _ in chan2emg[2]]
ax3.plot(x[chan_req], y[chan_req], "r.")
ax3.axis("off")

ax4 = fig.add_subplot(gs[3, 0])
ax4.plot(x, y, "k.", markersize=3)
chan_req = [np.where(chanprobe == _)[0][0] for _ in chan2emg[3]]
ax4.plot(x[chan_req], y[chan_req], "r.")
ax4.axis("off")


ax5 = fig.add_subplot(gs[4, 0])
ax5.plot(x, y, "k.", markersize=3)
chan_req = [np.where(chanprobe == _)[0][0] for _ in chan2emg[4]]
ax5.plot(x[chan_req], y[chan_req], "r.")
ax5.axis("off")


ax6 = fig.add_subplot(gs[0, 1:5])
ax6.plot(df["top"], color="#384052")

ax7 = fig.add_subplot(gs[1, 1:5])
ax7.plot(df["top_bottom"], color="#384052")

ax8 = fig.add_subplot(gs[2, 1:5])
ax8.plot(df["sup_deep"], color="#384052")

ax9 = fig.add_subplot(gs[3, 1:5])
ax9.plot(df["sup_middle_deep"], color="#384052")

ax10 = fig.add_subplot(gs[4, 1:5])
ax10.plot(df["all"], color="#384052")
ax10.set_xlabel("Time")
ax10.set_ylabel("Mean correlation")


ax11 = fig.add_subplot(gs[0, 5])
df.hist(column=["top"], bins=200, grid=False, ax=ax11, color="#f45d69")
ax11.set_title("")

ax12 = fig.add_subplot(gs[1, 5], sharex=ax11)
df.hist(column=["top_bottom"], bins=200, grid=False, ax=ax12, color="#f45d69")
ax12.set_title("")


ax13 = fig.add_subplot(gs[2, 5], sharex=ax11)
df.hist(column=["sup_deep"], bins=200, grid=False, ax=ax13, color="#f45d69")
ax13.set_title("")


ax14 = fig.add_subplot(gs[3, 5], sharex=ax11)
df.hist(column=["sup_middle_deep"], bins=200, grid=False, ax=ax14, color="#f45d69")
ax14.set_title("")


ax15 = fig.add_subplot(gs[4, 5], sharex=ax11)
df.hist(column=["all"], bins=200, grid=False, ax=ax15, color="#f45d69")
ax15.set_title("")
# ax1.axis("off")

fig.suptitle("Comparing correlation Emg for various electrode combination")
