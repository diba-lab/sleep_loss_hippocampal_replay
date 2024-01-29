#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import ipywidgets as widgets
import random

from callfunc import processData
import signal_process
from mathutil import threshPeriods


#%% ====== functions needed for some computation ============


def getPxx(lfp):
    window = 5 * 1250

    freq, Pxx = sg.welch(
        lfp, fs=1250, nperseg=window, noverlap=window / 6, detrend="linear",
    )
    noise = np.where(
        ((freq > 59) & (freq < 61)) | ((freq > 119) & (freq < 121)) | (freq > 220)
    )[0]
    freq = np.delete(freq, noise)
    Pxx = np.delete(Pxx, noise)

    return Pxx, freq


#%% Subjects
basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/"
]
sessions = [processData(_) for _ in basePath]

#%% Detects Low States during SD
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.2)

for sess in sessions:
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    post = sess.epochs.post
    localsleep = sess.localsleep.events.start

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    tstart, tend = post[0], post[0] + 5 * 3600
    lfpSD = lfp[(t > tstart) & (t < tend)]

    specgram = signal_process.spectrogramBands(
        lfpSD, window=eegSrate, overlap=0.9 * eegSrate
    )
    pxx = specgram.sxx
    freq = specgram.freq
    pxx_time = specgram.time

    # low states band
    lowState_freq = np.where((freq > 0.5) & (freq < 5))[0]
    pxx_lowState = pxx[lowState_freq, :]
    mean_pxx_lowState = np.mean(pxx_lowState, axis=0)
    zsc_lowState = stats.zscore(gaussian_filter1d(mean_pxx_lowState, sigma=0.5))
    zsc_t = np.linspace(tstart, tend, len(zsc_lowState))

    lfp_arnd_localsleep = [
        zsc_lowState[(zsc_t > t - 100) & (zsc_t < t + 100)] for t in localsleep
    ]

    ax = fig.add_subplot(gs[0])
    ax.plot(zsc_t, zsc_lowState)
    ax.plot(
        sess.localsleep.events.start, np.ones(len(sess.localsleep.events.start)), "."
    )


# endregion

# %% Detect lowstates during NREM episodes
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.subplots_adjust(hspace=0.2)

for sub, sess in enumerate(sessions):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    post = sess.epochs.post
    states = sess.brainstates.states
    nrem = states[(states["start"] > post[0]) & (states["name"] == "nrem")]

    lfp, _, _ = sess.ripple.best_chan_lfp()
    lfp = lfp[0, :]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfpNrem = []
    for epoch in nrem.itertuples():
        lfpNrem = lfp[(t > epoch.start) & (t < epoch.end)]
    lfpNrem = np.asarray(lfpNrem)

    specgram = signal_process.spectrogramBands(
        lfpNrem, window=eegSrate, overlap=0.1 * eegSrate
    )
    pxx = specgram.sxx
    freq = specgram.freq
    pxx_time = specgram.time

    # low states band
    lowState_freq = np.where((freq > 0.5) & (freq < 50))[0]
    pxx_lowState = pxx[lowState_freq, :]
    mean_pxx_lowState = np.mean(pxx_lowState, axis=0)
    zsc_lowState = stats.zscore(gaussian_filter1d(mean_pxx_lowState, sigma=0.5))
    histlfp, edges = np.histogram(zsc_lowState, bins=np.linspace(-2, 2, 100))
    # zsc_t = np.linspace(tstart, tend, len(zsc_lowState))

    subname = sess.sessinfo.session.sessionName
    ax = fig.add_subplot(gs[sub])
    ax.plot(edges[:-1], histlfp)
    ax.set_title(subname)
# endregion


# %%
