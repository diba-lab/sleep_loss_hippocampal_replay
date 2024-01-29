import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import linecache

from mpl_toolkits.mplot3d import Axes3D


# filename = '/data/Clustering/SleepDeprivation/RatJ/Behavior_Position/Take 2019-05-31 03.55.25 AM.fbx'
filename = (
    "/data/Clustering/SleepDeprivation/RatK/Position/Take 2019-08-16 04.42.07 AM.fbx"
)

fid = open(filename, "r")
# data = pd.read_csv(filename, header=5, skipfooter=800)

# time = data['Time (Seconds)'].tolist()
# x = data['X'].tolist()
# y = data['Y'].tolist()
# z = data['Z'].tolist()


# plt.plot(x, z, '.')

# flines = f.readlines()
# i = 1
k = 830
xpos, ypos, zpos = [], [], []
with open(filename) as f:
    next(f)
    for i, line in enumerate(f):

        m = "".join(line)

        if "RawSegs" in m:
            track_begin = i + 3
            line_frame = linecache.getline(filename, i + 3).strip().split(",")
            total_frames = float(line_frame[3])
            break


f.close()

with open(filename) as f:

    for i, line in enumerate(f):

        if i > track_begin:
            line = line.strip()
            m = line.split(",")

            xpos.append(m[1])
            ypos.append(m[2])
            zpos.append(m[3])

        if i == track_begin + total_frames - 1:
            break
f.close()

xpos = list(map(float, xpos))
ypos = list(map(float, ypos))
zpos = list(map(float, zpos))

plt.clf()
ax = plt.axes(projection="3d")
ax.plot3D(xpos, ypos, zpos)
