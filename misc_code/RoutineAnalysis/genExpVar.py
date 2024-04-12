import numpy as np
import os
from pathlib import Path as pth
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from spikesUtil import ExtractSpikes as getspk
from bokeh.plotting import figure, output_file, show, save


basePath = "/data/Clustering/SleepDeprivation/RatJ/Day1/"
sess1 = getspk(basePath)
sess1.CollectSpikes()
sess1.smoothEV()

fileInitial = sess1.filePrefix
expVar = np.load(fileInitial + "_EV.npy")
rev = np.load(fileInitial + "_REV.npy")
# states = np.load(fileInitial + "_behavior.npy")
epochs = np.load(fileInitial + "_epochs.npy", allow_pickle=True)

pre = epochs.item().get("PRE")  # in seconds
maze = epochs.item().get("MAZE")  # in seconds
post = epochs.item().get("POST")  # in seconds

t = np.linspace(post[0], post[1], len(expVar))


output_file("Expvar.html")

p = figure(
    title="POST (RatJDay1-SD)",
    plot_width=700,
    plot_height=400,
    x_axis_label="Time (h)",
    y_axis_label="Explained variance",
)

# add a line renderer
p.line(t / 3600, expVar, line_width=2, color="#605757", legend_label="EV")
p.line(t / 3600, rev, line_width=2, color="#75c791", legend_label="REV")
p.line([9, 9], [0.02, 0.12], line_width=2, color="#e89c9c", legend_label="SD stopped")

p.toolbar.logo = None

show(p)
