import numpy as np
import matplotlib.pyplot as plt
import os
from sleepDetect import SleepScore
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
import pandas as pd
from bokeh.models import Legend, LegendItem
from bokeh.layouts import row, column
from bokeh.layouts import grid, gridplot, layout

figurepath = "/data/DataGen/figuresGen/SleepDeprivation/"
output_file(figurepath + "sleepdetect.html")

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

sessions = [SleepScore(_) for _ in basePath]


first_last_hour = []
delta_all_inc, delta_t_inc = [], []
for i, sess in enumerate(sessions):
    # sess.deltaStates()
    epochs = np.load(sess.filePrefix + "_epochs.npy", allow_pickle=True)
    pre = epochs.item().get("PRE")  # in seconds
    maze = epochs.item().get("MAZE")  # in seconds
    post = epochs.item().get("POST")  # in seconds

    states = sess.sws_time
    states_dur = np.diff(states, axis=1)
    states = np.hstack((states, states_dur))

    if i in [0, 2, 4]:
        states = states[(states[:, 0] > post[0] + 5 * 3600) & (states[:, 2] > 300), :]

    else:
        states = states[(states[:, 0] > post[0]) & (states[:, 2] > 300), :]

    all_sws = []
    t_within = []
    for st in states:
        ind = np.where((sess.delta_t > st[0]) & (sess.delta_t < st[1]))[0]

        inc_time = sess.delta_t[ind]
        raw_sws = sess.delta[ind]
        all_sws.extend(raw_sws)
        x = np.linspace(0, 1, len(raw_sws))
        t_within.extend(inc_time)

        # z = np.polyfit(x, raw_sws, 1)
        # z_fit = z[0] * x + z[1]
        # plt.plot(x, z_fit)

    # x = np.linspace(0, 1, 100)
    # z = np.polyfit(t_within, all_sws, 1)
    # z_fit = z[0] * x + z[1]
    delta_all_inc.append(all_sws)
    delta_t_inc.append(t_within)

    first_last_hour.append(
        [np.mean(all_sws[:3600]), np.mean(all_sws[-3600:]), sess.subname]
    )


data = pd.DataFrame(first_last_hour, columns=["first", "last", "name"])

colormap = {"Day1": "red", "Day2": "green"}
colors = [colormap[x.split("_")[1]] for x in data["name"]]
p = figure(
    plot_width=400,
    plot_height=400,
    x_axis_label="",
    y_axis_label="Sws amplitude",
    x_range=(0, 3),
)

r = p.multi_line(
    [[1, 2] for _ in range(len(sessions))],
    [first_last_hour[x][:2] for x in range(6)],
    line_width=2,
    color=colors,
)
p.circle(np.ones(6), data["first"], color=colors, fill_alpha=0.2, size=10)
p.circle(2 * np.ones(6), data["last"], color=colors, fill_alpha=0.2, size=10)
p.xaxis.ticker = [1, 2]
p.xaxis.major_label_overrides = {1: "first hour", 2: "last hour"}
legend = Legend(
    items=[
        LegendItem(label="sd", renderers=[r], index=0),
        LegendItem(label="nsd", renderers=[r], index=1),
    ]
)
p.add_layout(legend)


p2 = figure(
    plot_width=700,
    plot_height=200,
    x_axis_label="",
    y_axis_label="Sws amplitude",
    x_range=(delta_t_inc[0][0] / 3600, delta_t_inc[0][-1] / 3600),
    y_range=(0, 20),
)

p2.line(
    sessions[0].delta_t / 3600,
    sessions[0].deltaraw,
    color="#c3c1c1",
    legend_label="raw amp",
)
p2.line(
    np.asarray(delta_t_inc[0]) / 3600,
    delta_all_inc[0],
    color="red",
    legend_label="smoothed amp",
)

p3 = figure(
    plot_width=700,
    plot_height=200,
    x_axis_label="Time (h)",
    y_axis_label="Sws amplitude",
    x_range=(delta_t_inc[5][0] / 3600, delta_t_inc[5][-1] / 3600),
    y_range=(0, 20),
)
p3.line(
    sessions[5].delta_t / 3600,
    sessions[5].deltaraw,
    color="#c3c1c1",
    legend_label="raw amp",
)
p3.line(
    np.asarray(delta_t_inc[5]) / 3600,
    delta_all_inc[5],
    color="green",
    legend_label="smoothed amp",
)
p.toolbar.logo = None
p2.toolbar.logo = None
p3.toolbar.logo = None
show(row(p, column(p2, p3)))

