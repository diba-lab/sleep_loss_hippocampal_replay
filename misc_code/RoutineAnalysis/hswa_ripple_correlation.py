import numpy as np
import matplotlib.pyplot as plt
from pfPlot import pf
import pandas as pd
import seaborn as sns
import altair as alt
import scipy.signal as sg
from numpy.fft import fft
import scipy.ndimage as filtSig
import matplotlib
from eventCorr import event_event

alt.renderers.enable("html")

# alt.data_transformers.enable("json")

basePath = [
    "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatK/Day1/",
    "/data/Clustering/SleepDeprivation/RatK/Day2/",
    "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

sessions = [event_event(_) for _ in basePath]


for sess_id, sess in enumerate(sessions):
    # sess.hswa()
    sess.hswa_ripple()

# chart = alt.hconcat()

chart = []
for sess_id, sess in enumerate(sessions):
    histdata = sess.hswa_ripple_hist

    chart.append(
        alt.Chart(histdata)
        .mark_line()
        .encode(
            x="time",
            y="swrs",
            color=alt.Color(
                "quant", scale=alt.Scale(scheme="redblue"), sort="descending"
            ),
            tooltip=["swrs"],
        )
        .interactive()
    )

# chart.facet(row=2)
p = alt.concat(*chart, columns=3)

p.save("filename.html")

