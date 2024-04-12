import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from callfunc import processData

# mpl.style.use("figPublish")


# TODO thoughts on using data class for loading data into function

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]


sessions_pre = [processData(_) for _ in basePath]
sessions_post = [processData(_) for _ in basePath]

# compute section

for sess_pre, sess_post in zip(sessions_pre, sessions_post):

    #%% --- Ripple power block -------------

    sess_pre.trange = sess_pre.epochs.pre
    sess_pre.eventpsth.hswa_ripple.nQuantiles = 5
    sess_pre.eventpsth.hswa_ripple.compute()

    sess_post.trange = np.asarray(
        [sess_post.epochs.post[0] + 5 * 3600, sess_post.epochs.post[1]]
    )
    sess_post.eventpsth.hswa_ripple.nQuantiles = 5
    sess_post.eventpsth.hswa_ripple.compute()


# plotting section

plt.clf()

fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(6, 2, figure=fig)


for sub, (sess_pre, sess_post) in enumerate(zip(sessions_pre, sessions_post)):

    #%% --- Ripple power block -------------

    ax1 = fig.add_subplot(gs[sub, 0])
    sess_pre.eventpsth.hswa_ripple.plot_ripplePower(ax1)

    ax2 = fig.add_subplot(gs[sub, 1])
    sess_post.eventpsth.hswa_ripple.plot_ripplePower(ax2)

    #%%===== Ripple peakpower over time ================

    # sess.trange = sess.epochs.pre
    # ripp_power = sess.ripple.peakpower
    # ripp_time = sess.ripple.time[:, 0]

    # ax1 = fig.add_subplot(gs[i, 0])
    # ax1.plot(ripp_time / 3600, ripp_power, ".", markersize=0.8, color="#cd7070")
    # ax1.set_title(sess.sessinfo.session.sessionName, x=0.2, y=0.90)
    # ax1.set_ylabel("peakpower")

    # sess.trange = np.asarray([sess.epochs.post[0] + 5 * 3600, sess.epochs.post[1]])
    # ripp_power = sess.ripple.peakpower
    # ripp_time = sess.ripple.time[:, 0]
    # ax2 = fig.add_subplot(gs[i, 1])
    # ax2.plot(ripp_time / 3600, ripp_power, ".", markersize=0.8, color="#83ce89")

    # fig.append(temp)

# ax1.set_xlabel("Time (h)")
# ax2.set_xlabel("Time (h)")

# fig.show()
