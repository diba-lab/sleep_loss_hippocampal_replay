import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import linecache

from mpl_toolkits.mplot3d import Axes3D


# filename = '/data/Clustering/SleepDeprivation/RatJ/Behavior_Position/Take 2019-05-31 03.55.25 AM.fbx'
filename = "/data/Clustering/SleepDeprivation/RatJ/Behavior_Position/Take 2019-06-02 03.59.05 AM_r.fbx"

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

        if "KeyCount" in m:
            track_begin = i + 2
            line_frame = linecache.getline(filename, i + 2).strip().split(" ")
            total_frames = int(line_frame[1]) - 1
            break


f.close()

with open(filename) as f:
    for _ in range(track_begin):
        next(f)

    for i, line in enumerate(f):
        # print(line)
        if len(xpos) > total_frames:
            break

        elif i < 1:
            print(i)
            line = line.strip()
            m = line.split(",")
            pos1 = m[1::5]
            print(pos1)

        else:
            line = line.strip()
            m = line.split(",")
            pos1 = m[2::5]

        xpos.extend(pos1)

    for _ in range(5):
        next(f)

    for i, line in enumerate(f):
        # print(line)
        if len(ypos) > total_frames:
            break

        elif i < 1:
            print(i)
            line = line.strip()
            m = line.split(",")
            pos1 = m[1::5]
            print(pos1)

        else:
            line = line.strip()
            m = line.split(",")
            pos1 = m[2::5]

        ypos.extend(pos1)

    for _ in range(5):
        next(f)

    for i, line in enumerate(f):
        # print(line)
        if len(zpos) > total_frames:
            break

        elif i < 1:
            print(i)
            line = line.strip()
            m = line.split(",")
            pos1 = m[1::5]
            print(pos1)

        else:
            line = line.strip()
            m = line.split(",")
            pos1 = m[2::5]

        # line = next(f)
        zpos.extend(pos1)

f.close()

