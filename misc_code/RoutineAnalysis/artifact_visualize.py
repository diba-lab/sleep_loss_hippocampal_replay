import numpy as np
import matplotlib.pyplot as plt
from artifactDetect import findartifact
import altair as alt
import pandas as pd


# alt.renderers.enable("html")

alt.data_transformers.enable("json")

basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    "/data/Clustering/SleepDeprivation/RatJ/Day3/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day4/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    # "/data/Clustering/SleepDeprivation/RatN/Day2/",
]

sessions = [findartifact(_) for _ in basePath]


for sess_id, sess in enumerate(sessions):
    # sess.hswa()
    sess.gen_artifact_epoch()


time = np.linspace(0, len(sess.zsc_signal) / 1250, len(sess.zsc_signal))
x = sess.zsc_signal

data = pd.DataFrame({"time": time, "sig": x})

# chart = alt.hconcat()
# plt.plot(sess.zsc_signal)
