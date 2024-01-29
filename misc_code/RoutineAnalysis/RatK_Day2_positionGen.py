import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime, timedelta
import pandas as pd
from getPosition import posfromFBX, getStartTime
import linecache


basePath = Path("/data/Clustering/SleepDeprivation/RatK/Day2/")
for file in os.listdir(basePath):
    if file.endswith(".eeg"):

        subname = file[:-4]
        filename = os.path.join(basePath, file)
        filePrefix = os.path.join(basePath, file[:-4])

# nframes = np.load(basePath / "og_files" / "numframes_OG.npy")
part1_frames = np.load(
    basePath / "og_files" / "2019-08-08_04-00-00_timestamps.npy", mmap_mode="r"
)
part2_frames = np.load(
    basePath / "og_files" / "2019-08-08_06-43-08_timestamps.npy", mmap_mode="r"
)

nframes1 = len(part1_frames)
nframes2 = len(part2_frames)

X1, Y1, Z1 = posfromFBX(basePath / "position" / "Take 2019-08-08 03.59.24 AM.fbx")
vid_start1 = getStartTime(basePath / "position" / "Take 2019-08-08 03.59.24 AM.csv")
vid_dur1 = len(X1) / (120 * 3600)
vid_end1 = vid_start1 + timedelta(hours=vid_dur1)


X2, Y2, Z2 = posfromFBX(basePath / "position" / "Take 2019-08-08 06.37.48 AM.fbx")
vid_start2 = datetime.strptime("2019-08-08 06.37.48 AM", "%Y-%m-%d %H.%M.%S %p")
vid_dur2 = len(X2) / (120 * 3600)
vid_end2 = vid_start2 + timedelta(hours=vid_dur2)

ephys_start1 = datetime.strptime("2019-08-08_04-00-00 AM", "%Y-%m-%d_%H-%M-%S %p")
ephys_dur1 = nframes1 / (3600 * 1250)
ephys_end1 = ephys_start1 + timedelta(hours=ephys_dur1)

ephys_start2 = datetime.strptime("2019-08-08_06-43-08 AM", "%Y-%m-%d_%H-%M-%S %p")
ephys_dur2 = nframes2 / (3600 * 1250)
ephys_end2 = ephys_start2 + timedelta(hours=ephys_dur2)


time_diff_start1 = ephys_start1 - vid_start1
time_diff_end1 = vid_end1 - ephys_end1

time_diff_start2 = ephys_start2 - vid_start2
time_diff_end2 = vid_end2 - ephys_end2


t_video1 = np.linspace(
    -time_diff_start1.total_seconds(),
    nframes1 / 1250 + time_diff_end1.total_seconds(),
    len(X1),
)
t_video2 = np.linspace(
    -time_diff_start2.total_seconds(),
    nframes2 / 1250 + time_diff_end2.total_seconds(),
    len(X2),
)

t_video_outside1 = np.argwhere((nframes1 / 1250 < t_video1) | (t_video1 < 0))
t_video_outside2 = np.argwhere((nframes2 / 1250 < t_video2) | (t_video2 < 0))

ind_keep1 = np.setdiff1d(np.arange(1, len(X1)), t_video_outside1)
ind_keep2 = np.setdiff1d(np.arange(1, len(X2)), t_video_outside2)

X1 = X1[ind_keep1]
X2 = X2[ind_keep2]
Z1 = Z1[ind_keep1]
Z2 = Z2[ind_keep2]

X = np.concatenate((X1, X2))
Z = np.concatenate((Z1, Z2))

posVar = {}
posVar["X"] = X
posVar["Y"] = Z
posVar["time"] = np.arange(0, len(X)) / 120

np.save(filePrefix + "_position.npy", posVar)

