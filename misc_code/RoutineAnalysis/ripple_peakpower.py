import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from callfunc import processData

# mpl.style.use("figPublish")


# TODO thoughts on using data class for loading data into function

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions = [processData(_) for _ in basePath]

# compute section

# for sess_pre, sess_post in zip(sessions_pre, sessions_rec, sessions_rec):

#     #%% --- Ripple power block -------------

#     sess_pre.trange = sess_pre.epochs.pre
#     sess_sd.trange = np.asarray([sess_post.epochs.post[0], sess_post.epochs.post[1]])
#     sess_rec.trange = np.asarray([sess_post.epochs.post[0], sess_post.epochs.post[1]])


# plotting section

plt.clf()

fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(6, 1, figure=fig)
fig.subplots_adjust(hspace=0.4)

for sub, sess in enumerate(sessions):

    #%% --- Ripple power block -------------

    ax1 = fig.add_subplot(gs[sub, 0])
    sess.trange = np.array([])
    power = sess.ripple.peakpower
    time = sess.ripple.time[:, 0]

    start = sess.epochs.pre[0]
    end = sess.epochs.post[1]
    mean_power_wind, mean_power_t, std_power_wind = [], [], []

    binsize = 10 * 60
    sldby = binsize
    for window in np.arange(start, end - binsize, sldby):
        indwhere = np.where((time > window) & (time < window + binsize))[0]
        # time_wind = time[indwhere]
        power_wind = power[indwhere]
        mean_power_wind.append(np.median(power_wind))
        std_power_wind.append(np.std(power_wind) / np.sqrt(len(indwhere)))
        mean_power_t.append((window + sldby / 2) / 3600)

    mean_power_t = np.asarray(mean_power_t)
    mean_power_wind = np.asarray(mean_power_wind)
    std_power_wind = np.asarray(std_power_wind)

    zt_time = mean_power_t - sess.epochs.post[0] / 3600
    zt_pre = (sess.epochs.pre - sess.epochs.post[0]) / 3600
    zt_post = (sess.epochs.post - sess.epochs.post[0]) / 3600

    ax1.fill_between(zt_pre, max(mean_power_wind), color="#c4c2bb")
    ax1.fill_between(zt_post, max(mean_power_wind), color="#eedc8b")
    if sub < 3:
        zt_sd = [0, 5]
        ax1.fill_between(zt_sd, max(mean_power_wind), color="#ecae7e")
        ax1.text(0, max(mean_power_wind) - 1, "SD", color="#ec3322", fontweight="bold")

    # ax2 = ax1.twinx()
    # ax2.plot((time - sess.epochs.post[0]) / 3600, power, ".", color="#a5d5a4")
    # ax2.set_ylim(0, max(power))

    ax1.plot(zt_time, mean_power_wind, color="k")
    ax1.fill_between(
        zt_time,
        mean_power_wind + std_power_wind,
        mean_power_wind - std_power_wind,
        color="k",
        alpha=0.3,
    )
    ax1.set_title(sess.sessinfo.session.sessionName, x=0.05, y=0.98, fontsize=10)
    ax1.set_ylim(min(mean_power_wind), max(mean_power_wind))
    ax1.set_xlim(min(zt_time), max(zt_time) + 0.5)

    # ax2 = fig.add_subplot(gs[sub, 1])
    # sess.trange = np.asarray([sess.epochs.post[0], sess.epochs.post[0] + 5 * 3600])
    # ripp_power = sess.ripple.peakpower
    # ripp_time = sess.ripple.time[:, 0]
    # ax2.plot(ripp_time / 3600, ripp_power, ".", markersize=0.8, color="#cd7070")

    # ax3 = fig.add_subplot(gs[sub, 2])
    # sess.trange = np.asarray([sess.epochs.post[0] + 5 * 3600, sess.epochs.post[1]])
    # ripp_power = sess.ripple.peakpower
    # ripp_time = sess.ripple.time[:, 0]
    # ax3.plot(ripp_time / 3600, ripp_power, ".", markersize=0.8, color="#cd7070")
ax1.set_xlabel("ZT time")
ax1.set_ylabel("Mean ripple amplitude")
fig.suptitle(
    "Ripple amplitude increases over the course of sleep deprivation (mean $\pm$ SEM, window = 10 min, slideby = 5 min)",
    fontsize=12,
)
# ax2.set_xlabel("Time (h)")

# fig.show()
