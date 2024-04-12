#%%
import os
import warnings

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import signal_process
from plotUtil import Fig
import subjects
import time
from joblib import Parallel, delayed


# warnings.simplefilter(action="default")


#%% functions
# region
def scale(x):

    x = x - np.min(x)
    x = x / np.max(x)

    return x


def getPxx(lfp):
    window = 5 * 1250

    freq, Pxx = sg.welch(
        lfp,
        fs=1250,
        nperseg=window,
        noverlap=window / 6,
        detrend="linear",
    )
    noise = np.where(
        ((freq > 59) & (freq < 61)) | ((freq > 119) & (freq < 121)) | (freq > 220)
    )[0]
    freq = np.delete(freq, noise)
    Pxx = np.delete(Pxx, noise)

    return Pxx, freq


# endregion

#%% Detect sleep states
# region
sessions = subjects.Nsd().ratSday2
for sess in sessions:
    # sess.brainstates.detect(emgfile=True)
    ax = sess.brainstates.hypnogram()
# endregion

#%% Spectrogram only
# region
figure = Fig()
fig, gs = figure.draw(grid=[4, 4])

axstate = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[0, :], hspace=0.2)
# sessions = subjects.Of().ratKday4
sessions = subjects.Tn().ratSday5
for sub, sess in enumerate(sessions):
    axspec = fig.add_subplot(axstate[1:4])
    chan = 113
    sess.viewdata.specgram(chan=chan, ax=axspec, window=10, overlap=2)
    axspec.axes.get_xaxis().set_visible(False)

# figure.savefig("spectrogram_example_sd", __file__)
# endregion

#%% Spectrogram example for figure panel
# region
figure = Fig()
fig, gs = figure.draw(num=1, grid=[8, 3])

sessions = subjects.Nsd().ratNday2 + subjects.Sd().ratNday1
for sub, sess in enumerate(sessions):
    # t_start = sess.epochs.post[0] + 5 * 3600
    t = sess.brainstates.params.time
    emg = sess.brainstates.params.emg
    delta = sess.brainstates.params.delta
    period = sess.epochs.post

    axstate = gridspec.GridSpecFromSubplotSpec(
        5, 1, subplot_spec=gs[sub, :2], hspace=0.2
    )
    axspec = fig.add_subplot(axstate[1:])
    sess.viewdata.specgram(ax=axspec, period=period)
    # axspec.axes.get_xaxis().set_visible(False)

    # axdelta = fig.add_subplot(axstate[4], sharex=axspec)
    # axdelta.fill_between(t, 0, delta, color="#9E9E9E")
    # axdelta.axes.get_xaxis().set_visible(False)
    # axdelta.set_ylabel("Delta")

    axhypno = fig.add_subplot(axstate[0], sharex=axspec)
    sess.brainstates.hypnogram(ax=axhypno, tstart=period[0])
    # sess.brainstates.addBackgroundtoPlots(ax=axhypno, tstart=period[0])
    figure.panel_label(axhypno, "a")

    # axemg = fig.add_subplot(axstate[-1], sharex=axspec)
    # axemg.plot(t, emg, "#4a4e68", lw=0.8)
    # axemg.set_ylabel("EMG")

# figure.savefig("spectrogram_example_sd_nsd", __file__)
# endregion

#%% EMG from lfp compare in case of 2 probes
# region
sessions = subjects.sd([3])
for sess in sessions:

    chan1 = np.concatenate(sess.recinfo.goodchangrp[:6]).astype(int)
    chan2 = np.concatenate(sess.recinfo.goodchangrp[6:]).astype(int)

    eegdata = sess.recinfo.geteeg(chans=0)
    total_duration = len(eegdata) / 1250
    window, overlap = 1, 0.2
    timepoints = np.arange(0, total_duration - window, window - overlap)

    def corrchan(chans, start):
        lfp_req = np.asarray(
            sess.recinfo.geteeg(chans=chans, timeRange=[start, start + window])
        )
        yf = signal_process.filter_sig.bandpass(lfp_req, lf=300, hf=600, fs=1250)
        ltriang = np.tril_indices(len(chans), k=-1)
        return np.corrcoef(yf)[ltriang].mean()

    corr_chan1 = Parallel(n_jobs=8, require="sharedmem")(
        delayed(corrchan)(chan1, start) for start in timepoints
    )
    corr_chan2 = Parallel(n_jobs=8, require="sharedmem")(
        delayed(corrchan)(chan2, start) for start in timepoints
    )

    corr_chan1 = gaussian_filter1d(corr_chan1, sigma=10)
    corr_chan2 = gaussian_filter1d(corr_chan2, sigma=10)

    plt.plot(corr_chan1)
    plt.plot(corr_chan2)


# endregion

#%% NREM,REM Duration compare between SD and control
# region
figure = Fig()
fig, gs = figure.draw(grid=[4, 4])
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    df = sess.brainstates.states
    df = df.loc[
        ((df.name == "rem") | (df.name == "nrem")) & (df.start > t_start)
    ].copy()
    df["condition"] = [condition] * len(df)
    group.append(df)

group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 0])
ax.clear()
sns.boxplot(x="state", y="duration", hue="condition", data=group, palette="Set3", ax=ax)
ax.set_ylim(-10, 2000)
ax.set_ylabel("duration (s)")
ax.set_xlabel("")
ax.set_xticklabels(["nrem", "rem"])
figure.panel_label(ax, "b")
figure.savefig("nrem_rem_duration_compare", __file__)
# endregion

#%% Mean theta-delta ratio compare between SD and control
# region
# plt.clf()
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    states = sess.brainstates.states
    states = states.loc[(states.name == "rem") & (states.start > t_start)].copy()

    params = sess.brainstates.params
    theta_delta = []
    for epoch in states.itertuples():
        val = params.loc[
            (params.time > epoch.start) & (params.time < epoch.end),
            "theta_deltaplus_ratio",
        ]
        theta_delta.append(np.mean(val))

    states.loc[:, "theta_delta"] = theta_delta
    states.loc[:, "condition"] = [condition] * len(states)
    group.append(states)


group = pd.concat(group, ignore_index=True)

figure = Fig()
fig, gs = figure.draw(grid=[2, 2])
ax = fig.add_subplot(gs[1, 1])
ax.clear()
sns.boxplot(
    x="condition", y="theta_delta", data=group, palette="Set3", ax=ax, width=0.5
)
ax.set_ylabel("ratio")
ax.set_xlabel("")
ax.set_title("Mean theta-delta ratio \n during REM")
figure.panel_label(ax, "c")

# endregion


#%% Delta amplitude 1st NREM
# region
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    states = sess.brainstates.states
    states = states.loc[
        (states.name == "nrem") & (states.start > t_start) & (states.duration > 300)
    ].copy()
    states["condition"] = [condition] * len(states)

    params = sess.brainstates.params
    first_nrem = states[:1].reset_index()
    val = params.loc[
        (params["time"] > first_nrem.start[0]) & (params["time"] < first_nrem.end[0]),
        "delta",
    ].copy()
    first_nrem["delta"] = np.mean(val)
    group.append(first_nrem)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 2])
ax.clear()
sns.boxplot(x="condition", y="delta", data=group, palette="Set3", ax=ax, width=0.5)
ax.set_ylabel("amplitude")
ax.set_xlabel("")
ax.set_title("Delta Power 1st NREM \n (>5 minutes)", fontsize=titlesize)
# ax.set_xticklabels(["nrem"])
panel_label(ax, "d")
# endregion

#%% Ripple band amplitude 1st NREM
# region
group = []
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    if sub < 3:
        t_start = sess.epochs.post[0] + 5 * 3600
        condition = "SD"
    else:
        t_start = sess.epochs.post[0]
        condition = "NSD"

    states = sess.brainstates.states
    states = states.loc[
        (states["name"] == "nrem")
        & (states["start"] > t_start)
        & (states["duration"] > 300)
    ].copy()
    states["condition"] = [condition] * len(states)

    params = sess.brainstates.params
    first_nrem = states[:1].reset_index()
    val = params.loc[
        (params["time"] > first_nrem.start[0]) & (params["time"] < first_nrem.end[0]),
        "ripple",
    ]
    first_nrem["ripple"] = np.mean(val)
    group.append(first_nrem)


group = pd.concat(group, ignore_index=True)
ax = fig.add_subplot(gs[1, 3])
ax.clear()
sns.boxplot(x="condition", y="ripple", data=group, palette="Set3", ax=ax, width=0.5)
ax.set_ylabel("amplitude")
ax.set_xlabel("")
ax.set_title("Ripple Power 1st NREM \n (>5 minutes)", fontsize=titlesize)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
panel_label(ax, "e")

# ax.set_xticklabels(["nrem"])
# endregion

#%% PSD first hour vs last hour and plotting the difference
# region
"""We want to understand the changes in spectral power across sleep deprivation, one interesting way to look at that is plotting the difference of power across frequencies.
"""
psd1st_all, psd5th_all, psd_diff = [], [], []
for sub, sess in enumerate(sessions[:3]):
    sess.trange = np.array([])
    eegSrate = sess.recinfo.lfpSrate
    channel = sess.theta.bestchan
    post = sess.epochs.post
    eeg = sess.utils.geteeg(chans=channel, timeRange=[post[0], post[0] + 5 * 3600])
    nfrms_hour = eegSrate * 3600
    lfp1st = eeg[:nfrms_hour]
    lfp5th = eeg[-nfrms_hour:]

    psd = lambda sig: sg.welch(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )
    multitaper = lambda sig: signal_process.mtspect(
        sig, fs=eegSrate, nperseg=10 * eegSrate, noverlap=5 * eegSrate
    )

    _, psd1st = multitaper(lfp1st)
    f, psd5th = multitaper(lfp5th)

    psd1st_all.append(psd1st)
    psd5th_all.append(psd5th)
    psd_diff.append(psd1st - psd5th)

psd1st_all = np.asarray(psd1st_all).mean(axis=0)
psd5th_all = np.asarray(psd5th_all).mean(axis=0)
psd_diff = np.asarray(psd_diff).mean(axis=0)

ax = fig.add_subplot(gs[2, 1])
ax.clear()
# ax.plot(f, psd1st_all, "k", label="ZT1")
# ax.plot(f, psd5th_all, "r", label="ZT5")
ax.plot(stats.zscore(psd_diff), "k")
ax.set_xscale("log")
ax.set_xlim([1, 300])
# ax.set_ylim([10, 10e5])
ax.set_title("PSDZT1-PSDZT5", fontsize=titlesize)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Zscore of difference")

# TODO add distributions of emg changes across the SD period
# ax.legend()
panel_label(ax, "f")

# endregion


#%% Powers at various bands scross sleep deprivation
# region
group = []
for sub, sess in enumerate(sessions[:3]):

    sess.trange = np.array([])
    lfp = sess.spindle.best_chan_lfp()[0]
    eegSrate = sess.recinfo.lfpSrate
    post = sess.epochs.post
    sd_period = [post[0], post[0] + 5 * 3600]
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfpsd = lfp[(t > sd_period[0]) & (t < sd_period[1])]
    tsd = np.linspace(sd_period[0], sd_period[1], len(lfpsd))
    binsd = np.linspace(sd_period[0], sd_period[1], 6)

    specgram = signal_process.spectrogramBands(lfpsd)
    bands = [
        specgram.delta,
        specgram.theta,
        specgram.spindle,
        specgram.gamma,
        specgram.ripple,
    ]

    mean_bands = stats.binned_statistic(
        specgram.time + sd_period[0], bands, statistic="mean", bins=binsd
    )

    mean_bands = mean_bands.statistic.T / np.sum(mean_bands.statistic, axis=1)

    df = pd.DataFrame(
        mean_bands, columns=["delta", "theta", "spindle", "gamma", "ripple"]
    )
    subname = sess.sessinfo.session.sessionName
    df["subject"] = [subname] * len(df)
    df["hour"] = np.arange(1, 6)
    group.append(df)


group = pd.concat(group, ignore_index=True)
group_long = pd.melt(
    group, id_vars=["hour", "subject"], var_name=["bands"], value_name="amplitude"
)

cmap = mpl.cm.get_cmap("Set3")
colors = [cmap(ind) for ind in range(5)]
colors = np.asarray(list(np.concatenate([[col] * 5 for col in colors])))
ax = fig.add_subplot(gs[2, 2:4])
ax.clear()
sns.barplot(
    x="bands",
    y="amplitude",
    hue="hour",
    data=group_long,
    # palette="Set3",
    color=colors[0],
    # edgecolor=".05",
    errwidth=1,
    # ax=ax,
    ci="sd",
)
ax.set_ylabel("Normalized amplitude")
# ax.legend(ncol=5)
ax.legend("")
ax.set_xlabel("")
ax.set_title("Band power during SD (hourly, 5 hours)", fontsize=titlesize)
fig.show()
panel_label(ax, "g")


scriptname = os.path.basename(__file__)
filename = "Test"
savefig(fig, filename, scriptname)
# endregion


#%%* REM Powerspectrum compare between SD and control session
# region
figure = Fig()
fig, gs = figure.draw(grid=[4, 4])
axrem = plt.subplot(gs[3, 0])
figure.panel_label(axrem, "d")

pxx = pd.DataFrame()
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    sampfreq = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    tstart = sess.epochs.post[0]
    deadtime = sess.artifact.time

    lfp = sess.theta.getBestChanlfp()
    t = np.linspace(0, len(lfp) / sampfreq, len(lfp))

    # ---- getting maze lfp ----------
    lfpmaze = sess.utils.geteeg(sess.theta.bestchan, timeRange=maze)

    # --- getting rem episodes lfp --------
    states = sess.brainstates.states
    rem = states[(states["start"] > tstart) & (states["name"] == "rem")]
    remframes = np.concatenate(
        [
            np.arange(int(epoch.start * sampfreq), int(epoch.end * sampfreq))
            for epoch in rem.itertuples()
        ]
    )
    lfprem = lfp[remframes]

    # ---- deleting noisy frames ------------
    if deadtime is not None:
        lfpmaze = sess.artifact.removefrom(lfpmaze, timepoints=maze)
        lfprem = sess.artifact.removefrom(lfprem, timepoints=remframes / sampfreq)

    # lfprem = stats.zscore(lfprem)
    # lfpmaze = stats.zscore(lfpmaze)
    if sub < 3:
        condition = "sd"
    else:
        condition = "nsd"

    pxx = pxx.append(
        pd.DataFrame(
            {
                "freq": getPxx(lfprem)[1],
                "maze": getPxx(lfpmaze)[0],
                "rem": getPxx(lfprem)[0],
                "condition": condition,
            },
        )
    )


# -----Plotting ---------
pxx["diff"] = pxx.rem - pxx.maze
pxx_mean = (
    pxx.groupby(["condition", "freq"])
    .mean()
    .transform(stats.zscore, axis=0)
    .reset_index()
)


sns.lineplot(
    x="freq",
    y="diff",
    hue="condition",
    data=pxx_mean,
    ax=axrem,
    palette="Set2",
    legend=None,
)
axrem.set_xlim([0, 20])
figure.savefig("rem_pxx_sd_vs_nsd", __file__)

# endregion

#%% Bicoherence plots of REM sleep
# region
plt.clf()
fig = plt.figure(1, figsize=(10, 15))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.subplots_adjust(hspace=0.3)
for sub, sess in enumerate(sessions):

    sess.trange = np.array([])

    sampfreq = sess.recinfo.lfpSrate
    maze = sess.epochs.maze
    tstart = sess.epochs.post[0]

    lfp, _, _ = sess.spindle.best_chan_lfp()
    t = np.linspace(0, len(lfp) / 1250, len(lfp))
    deadfile = sess.sessinfo.files.filePrefix.with_suffix(".dead")
    if deadfile.is_file():
        with deadfile.open("r") as f:
            noisy = []
            for line in f:
                epc = line.split(" ")
                epc = [float(_) for _ in epc]
                noisy.append(epc)
            noisy = np.asarray(noisy)
            noisy = ((noisy / 1000) * sampfreq).astype(int)

        for noisy_ind in range(noisy.shape[0]):
            st = noisy[noisy_ind, 0]
            en = noisy[noisy_ind, 1]
            numnoisy = en - st
            lfp[st:en] = np.nan

    states = sess.brainstates.states
    rem = states[(states["start"] > tstart) & (states["name"] == "rem")]

    binlfp = lambda x, t1, t2: x[(t > t1) & (t < t2)]
    lfprem = []
    for epoch in rem.itertuples():
        lfprem.extend(binlfp(lfp, epoch.start, epoch.end))
    lfprem = stats.zscore(np.asarray(lfprem))

    # strong_theta = strong_theta - np.mean(strong_theta)
    lfprem = sg.detrend(lfprem, type="linear")
    bicoh, bicoh_freq = signal_process.bicoherence(
        lfprem, window=4 * 1250, overlap=2 * 1250
    )

    ax = fig.add_subplot(gs[sub])
    ax.pcolorfast(bicoh_freq, bicoh_freq, bicoh, cmap="YlGn", vmax=0.2)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Frequency (Hz)")
    # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    ax.set_ylim([2, 75])

    # ax = fig.add_subplot(gs[sub + 2])
    # f, t, sxx = sg.spectrogram(strong_theta, nperseg=1250, noverlap=625, fs=1250)
    # ax.pcolorfast(t, f, sxx, cmap="YlGn", vmax=0.05)
    # ax.set_ylabel("Frequency (Hz)")
    # ax.set_xlabel("Time (s)")
    # # plt.pcolormesh(bispec_freq, bispec_freq, bispec, vmin=0, vmax=0.1, cmap="YlGn")
    # ax.set_ylim([1, 75])

fig.suptitle("fourier and bicoherence analysis of strong theta during MAZE")
# endregion


#%%* Sleep proportion --> sleep deprivation vs control
# region
sessions = subjects.Nsd().allsess + subjects.Sd().allsess
sd_prop, nsd_prop = [], []
for sub, sess in enumerate(sessions):
    eegSrate = sess.recinfo.lfpSrate
    tag = sess.recinfo.animal.tag
    post = sess.epochs.post

    period = [post[0], post[0] + 5 * 3600]
    states_prop = sess.brainstates.proportion(period=period)
    if tag == "sd":
        sd_prop.append(states_prop)
    else:
        nsd_prop.append(states_prop)

sd_prop = pd.concat(sd_prop).groupby("name").sum() / 4
nsd_prop = pd.concat(nsd_prop).groupby("name").sum() / 4

sd_prop = sd_prop.rename(columns={"duration": "sd"})
nsd_prop = nsd_prop.rename(columns={"duration": "nsd"})

all_grp = pd.concat([sd_prop, nsd_prop], axis=1)

figure = Fig()
fig, gs = figure.draw(num=1, grid=(4, 3))
ax = plt.subplot(gs[0])
all_grp.T.plot.bar(
    stacked=True, color=sess.brainstates.colors, alpha=0.5, ax=ax, legend=None, rot=45
)

ax.set_ylabel("Proportion")
ax.set_title("first 5 hours")

# figure.savefig("sleep_propotion", __file__)

# endregion
