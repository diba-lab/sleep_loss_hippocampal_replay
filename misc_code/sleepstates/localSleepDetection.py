#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib as mpl
import scipy.ndimage as smooth
import signal_process
from plotUtil import Fig
import subjects

#%% localsleep detection
# region
sessions = subjects.Sd().allsess
for sub, sess in enumerate(sessions):
    post = sess.epochs.post
    period = [post[0], post[0] + 5 * 3600]
    sess.localsleep.detect(period=period)
    # sess.localsleep.plot_examples()
# endregion

#%% Localsleep event rate across sleep deprivation
# region

sessions = subjects.Sd().allsess
data = pd.DataFrame()
for sub, sess in enumerate(sessions):
    post = sess.epochs.post
    lcslp = sess.localsleep.events
    period = [post[0], post[0] + 5 * 3600]
    bins = np.linspace(period[0], period[1], 6)
    counts = np.histogram(lcslp.start, bins=bins)[0]

    data = data.append(
        pd.DataFrame({"bins": np.arange(2), "counts": counts[[0, 4]] / 60, "name": sub})
    )

figure = Fig()
fig, gs = figure.draw(num=1, grid=(3, 3))
ax = plt.subplot(gs[0])
sns.pointplot(
    data=data,
    x="bins",
    y="counts",
    hue="name",
    # color="#c5c3c6",
    ax=ax,
    palette=["#c5c3c6"] * 5,
    zorder=1,
)
sns.pointplot(
    data=data, x="bins", y="counts", palette="gray", join=True, ax=ax, zorder=2
)
ax.set_ylim(bottom=0)


# endregion
#%% Local sleep example plots plus summary from all SD sessions
# region
plt.close("all")
# fig = plt.figure(1, figsize=(1, 15))
# gs = GridSpec(4, 3, figure=fig)
# fig.subplots_adjust(hspace=0.5)


for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    # sess.spikes.fromCircus("same_folder")
    # sess.localsleep.detect(period=[tstart, tend])

    if sub == 2:
        fig = sess.localsleep.plot()

    # fig = sess.localsleep.plot()
    # sess.localsleep.plotAll()

col = ["#FF8F00", "#388E3C", "#9C27B0"]

sd1 = np.zeros(3)
sd5 = np.zeros(3)
ax = fig.add_subplot(3, 5, 12)
for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    tbin_offperiods = np.linspace(tstart, tend, 6)
    t_offperiods = sess.localsleep.events.start.values
    hist_off = np.histogram(t_offperiods, bins=tbin_offperiods)[0]
    hist_off = hist_off / 60
    sd1[sub] = hist_off[0]
    sd5[sub] = hist_off[-1]
    # plt.plot(hist_off / 60)

colsub = "#9E9E9E"
ax = fig.add_subplot(3, 5, 11)
ax.plot(np.ones(3), sd1, "o", color=colsub)
ax.plot(3 * np.ones(3), sd5, "o", color=colsub)
ax.plot([1, 3], np.vstack((sd1, sd5)), color=colsub, linewidth=0.8)

mean_grp = np.array([np.mean(sd1), np.mean(sd5)])
sem_grp = np.array([stats.sem(sd1), stats.sem(sd5)])

ax.errorbar(np.array([1, 3]), mean_grp, yerr=sem_grp, color="#263238", fmt="o")
# ax.plot([1, 3], [np.mean(sd1), np.mean(sd5)], color="#263238")
ax.set_xlim([0, 4])
ax.set_ylim([5, 25])
ax.set_xticks([1, 3])
ax.set_xticklabels(["SD1", "SD5"])
ax.set_ylabel("Number per min")

# colsub = "#9E9E9E"
ax = fig.add_subplot(3, 5, 12)

for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    fbefore = sess.localsleep.instfiringbefore[:-1].mean(axis=0)
    fbeforestd = sess.localsleep.instfiringbefore[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    fafter = sess.localsleep.instfiringafter[:-1].mean(axis=0)
    fafterstd = sess.localsleep.instfiringafter[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    tbefore = np.linspace(-1, 0, len(fbefore))
    tafter = np.linspace(0.2, 1.2, len(fafter))

    # ax.fill_between(
    #     [0, 0.2],
    #     [min(fbefore), min(fbefore)],
    #     [max(fbefore), max(fbefore)],
    #     color="#BDBDBD",
    #     alpha=0.3,
    # )
    ax.fill_between(
        tbefore, fbefore + fbeforestd, fbefore - fbeforestd, color="#BDBDBD"
    )
    # ax.plot(tbefore, fbefore, color="#616161")
    ax.fill_between(tafter, fafter + fafterstd, fafter - fafterstd, color="#BDBDBD")
    # ax.plot(tafter, fafter, color="#616161")

    # self.events["duration"].plot.kde(ax=ax, color="k")
    # ax.set_xlim([0, max(self.events.duration)])
    ax.set_xlabel("Time from local sleep (s)")
    ax.set_ylabel("Instantneous firing")
    ax.set_xticks([-1, -0.5, 0, 0.2, 0.7, 1.2])
    ax.set_xticklabels(["-1", "-0.5", "start", "end", "0.5", "1"], rotation=45)


ax = fig.add_subplot(3, 5, 13)

for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    sess.localsleep.events.duration.plot.kde(color="#BDBDBD")

ax.set_xlim([0, 3])
ax.set_xlabel("Duration (s)")
# endregion


#%% Spectrogram, Raster plot and Local sleep example plots plus summary from all SD sessions
# region
plt.clf()
fig = plt.figure(num=None, figsize=(10, 15))
gs = gridspec.GridSpec(4, 5, figure=fig)
fig.subplots_adjust(hspace=0.5)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    spikes = sess.spikes.times
    if sub == 2:

        ax = fig.add_subplot(gs[0, :])
        sess.viewdata.specgram(ax=ax)
        ax.set_xlim([tstart, tend])

        ax = fig.add_subplot(gs[1, :], sharex=ax)
        sess.viewdata.raster(ax=ax, period=[tstart, tend])

        # ax = fig.add_subplot(gs[2, :])
        # sess.localsleep.plot(fig=fig, ax=ax)

    # fig = sess.localsleep.plot()
    # sess.localsleep.plotAll()

col = ["#FF8F00", "#388E3C", "#9C27B0"]

sd1 = np.zeros(3)
sd5 = np.zeros(3)
ax = fig.add_subplot(3, 5, 12)
for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    tbin_offperiods = np.linspace(tstart, tend, 6)
    t_offperiods = sess.localsleep.events.start.values
    hist_off = np.histogram(t_offperiods, bins=tbin_offperiods)[0]
    hist_off = hist_off / 60
    sd1[sub] = hist_off[0]
    sd5[sub] = hist_off[-1]
    # plt.plot(hist_off / 60)

colsub = "#9E9E9E"
ax = fig.add_subplot(3, 5, 11)
ax.plot(np.ones(3), sd1, "o", color=colsub)
ax.plot(3 * np.ones(3), sd5, "o", color=colsub)
ax.plot([1, 3], np.vstack((sd1, sd5)), color=colsub, linewidth=0.8)

mean_grp = np.array([np.mean(sd1), np.mean(sd5)])
sem_grp = np.array([stats.sem(sd1), stats.sem(sd5)])

ax.errorbar(np.array([1, 3]), mean_grp, yerr=sem_grp, color="#263238", fmt="o")
# ax.plot([1, 3], [np.mean(sd1), np.mean(sd5)], color="#263238")
ax.set_xlim([0, 4])
ax.set_ylim([5, 25])
ax.set_xticks([1, 3])
ax.set_xticklabels(["SD1", "SD5"])
ax.set_ylabel("Number per min")

# colsub = "#9E9E9E"
ax = fig.add_subplot(3, 5, 12)

for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600

    fbefore = sess.localsleep.instfiringbefore[:-1].mean(axis=0)
    fbeforestd = sess.localsleep.instfiringbefore[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    fafter = sess.localsleep.instfiringafter[:-1].mean(axis=0)
    fafterstd = sess.localsleep.instfiringafter[:-1].std(axis=0) / np.sqrt(
        len(sess.localsleep.events)
    )
    tbefore = np.linspace(-1, 0, len(fbefore))
    tafter = np.linspace(0.2, 1.2, len(fafter))

    # ax.fill_between(
    #     [0, 0.2],
    #     [min(fbefore), min(fbefore)],
    #     [max(fbefore), max(fbefore)],
    #     color="#BDBDBD",
    #     alpha=0.3,
    # )
    ax.fill_between(
        tbefore, fbefore + fbeforestd, fbefore - fbeforestd, color="#BDBDBD"
    )
    # ax.plot(tbefore, fbefore, color="#616161")
    ax.fill_between(tafter, fafter + fafterstd, fafter - fafterstd, color="#BDBDBD")
    # ax.plot(tafter, fafter, color="#616161")

    # self.events["duration"].plot.kde(ax=ax, color="k")
    # ax.set_xlim([0, max(self.events.duration)])
    ax.set_xlabel("Time from local sleep (s)")
    ax.set_ylabel("Instantneous firing")
    ax.set_xticks([-1, -0.5, 0, 0.2, 0.7, 1.2])
    ax.set_xticklabels(["-1", "-0.5", "start", "end", "0.5", "1"], rotation=45)


ax = fig.add_subplot(3, 5, 13)

for sub, sess in enumerate(sessions):
    tstart = sess.epochs.post[0]
    tend = sess.epochs.post[0] + 5 * 3600
    sess.localsleep.events.duration.plot.kde(color="#BDBDBD")

ax.set_xlim([0, 3])
ax.set_xlabel("Duration (s)")
# endregion


#%% localsleep and ripples around it
# region
plt.clf()
fig = plt.figure(num=1, figsize=(10, 15))
gs = gridspec.GridSpec(1, 1, figure=fig)
fig.subplots_adjust(hspace=0.5)
ax = fig.add_subplot(gs[0])
colors = ["#ff928a", "#424242", "#3bceac"]

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    locsleep = sess.localsleep.events
    ripples = sess.ripple.time

    tbin_before = lambda t: np.linspace(t - 1, t, 20)
    tbin_after = lambda t: np.linspace(t, t + 1, 20)
    ripples_counts_before = np.asarray(
        [
            np.histogram(ripples[:, 0], bins=tbin_before(event))[0]
            for event in locsleep.start
        ]
    )
    ripples_counts_after = np.asarray(
        [
            np.histogram(ripples[:, 0], bins=tbin_after(event))[0]
            for event in locsleep.end
        ]
    )

    total_ripples_before = np.sum(ripples_counts_before, axis=0)
    total_ripples_after = np.sum(ripples_counts_after, axis=0)

    combined = np.concatenate((total_ripples_before, total_ripples_after))

    subname = sess.sessinfo.session.sessionName
    ax.plot(
        np.linspace(-1, 0, 19),
        total_ripples_before,
        color=colors[sub],
        label=subname,
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        np.linspace(0.5, 1.5, 19),
        total_ripples_after,
        color=colors[sub],
        lw=2,
        alpha=0.8,
    )
    ax.set_xlabel("Time from localsleep (s)")
    ax.set_ylabel("# SWRs")
ax.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5])
ax.set_xticklabels([-1, -0.5, "start", "end", 1, 1.5])
ax.legend()
# endregion


#%% Spectrogram around localsleep
# region
plt.clf()
fig = plt.figure(num=1, figsize=(10, 15))
gs = gridspec.GridSpec(3, 1, figure=fig)
fig.subplots_adjust(hspace=0.2)

for sub, sess in enumerate(sessions):

    sess.trange = np.array([])
    locslp = sess.localsleep.events
    eegSrate = sess.recinfo.lfpSrate
    nShanks = sess.recinfo.nShanks
    changrp = sess.recinfo.channelgroups[3]
    lfp = np.asarray(sess.utils.geteeg(channels=changrp[-1]))

    # lfp, chan, _ = sess.spindle.best_chan_lfp()
    # print(chan)
    t = np.linspace(0, len(lfp) / eegSrate, len(lfp))

    lfp_locslp_ind = []
    for evt in locslp.itertuples():
        lfp_locslp_ind.extend(
            np.arange(int((evt.end - 0.1) * eegSrate), int((evt.end + 0.1) * eegSrate))
        )
    lfp_locslp_ind = np.asarray(lfp_locslp_ind)
    lfp_locslp = lfp[lfp_locslp_ind]
    lfp_locslp_avg = np.reshape(lfp_locslp, (len(locslp), 250)).mean(axis=0)
    t_locslp = np.linspace(0, len(lfp_locslp) / eegSrate, len(lfp_locslp))

    freqs = np.arange(20, 100, 0.5)
    wavdec = signal_process.wavelet_decomp(lfp_locslp, freqs=freqs)
    # wav = wavdec.cohen(ncycles=ncycles)
    wav = wavdec.cohen(ncycles=3)
    wav = (
        stats.zscore(wav, axis=1).reshape((wav.shape[0], 250, len(locslp))).mean(axis=2)
    )
    wav = smooth.gaussian_filter(wav, sigma=2)

    ax = fig.add_subplot(gs[sub])
    ax.pcolorfast(np.linspace(-100, 100, 250), freqs, wav, cmap="jet")
    ax2 = ax.twinx()
    ax2.plot(np.linspace(-100, 100, 250), lfp_locslp_avg, color="white")
    ax.spines["right"].set_visible(True)
    ax.set_xlabel("Time from start of localsleep periods (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim([-100, 100])


# endregion
