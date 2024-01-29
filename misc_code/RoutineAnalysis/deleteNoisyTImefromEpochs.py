import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime, timedelta
import pandas as pd

basePath = Path("/data/Clustering/SleepDeprivation/RatJ/Day1/")
for file in os.listdir(basePath):
    if file.endswith(".eeg"):

        subname = file[:-4]
        filename = os.path.join(basePath, file)
        filePrefix = os.path.join(basePath, file[:-4])

nframes = np.load(basePath / "og_files" / "numframes_OG.npy")
DatFileOG = basePath / "RatJ_Day1_2019-05-31_03-55-36.dat"
endFrametime = len(np.memmap(DatFileOG, dtype="int16", mode="r")) / (75 * 30000)

noisyFrames = np.load(basePath / "og_files" / "noisy_timestamps_fromOG.npy")
# epochs = np.load(
#     basePath / "og_files" / "RatJ_Day1_2019-05-31_03-55-36_epochs_og.npy",
#     allow_pickle=True,
# )
posInfo = np.load(
    basePath / "og_files" / "RatJ_Day1_2019-05-31_03-55-36_position_OG.npy",
    allow_pickle=True,
)

posX = np.asarray(posInfo.item().get("X"))
posY = np.asarray(posInfo.item().get("Y"))
frames = posInfo.item().get("frames")
video_starttime = posInfo.item().get("begin")
video_duration = len(posX) / (120 * 3600)
video_endtime = video_starttime + timedelta(hours=video_duration)


ephys_starttime = datetime.strptime(subname[-19:], "%Y-%m-%d_%H-%M-%S")

time_diff = ephys_starttime - video_starttime


ephys_duration = nframes / (3600 * 1250)
ephys_endtime = ephys_starttime + timedelta(hours=ephys_duration)
time_record = np.arange(ephys_starttime, ephys_endtime, dtype="datetime64[h]")
time_diff_end = video_endtime - ephys_endtime


# start = pd.Timestamp(ephys_starttime)
# end = pd.Timestamp(ephys_endtime)
# tim2 = pd.to_datetime(np.linspace(start.value, end.value, nframes))
# tim2 = np.asarray(tim2)
# plt.plot(posY)

t_ephys = np.arange(0, nframes) / 1250
t_video = np.linspace(
    -time_diff.total_seconds(),
    nframes / 1250 + time_diff_end.total_seconds(),
    len(posX),
)

noisy_time = noisyFrames / 1250
t_video_noisy = np.concatenate(
    [np.where(np.digitize(t_video, x) == 1)[0] for x in noisy_time], axis=0
)
t_video_outside = np.argwhere((nframes / 1250 < t_video) | (t_video < 0))
t_video_noisy = np.union1d(t_video_outside, t_video_noisy)
ind_good = np.setdiff1d(np.arange(1, len(posX)), t_video_noisy)
t_video_keep = t_video[ind_good]
posX_keep = posX[ind_good]
posY_keep = posY[ind_good]


posVar = {}
posVar["X"] = posX_keep
posVar["Y"] = posY_keep
posVar["time_orig"] = t_video_keep
posVar["time"] = np.arange(0, len(t_video_keep)) / 120
# posVar["begin"] = self.tbegin
# # posVar["pos_sRate"] = self.optitrack_sRate
np.save(Path(basePath, subname + "_position.npy"), posVar)
# k = np.load(Path(basePath, subname + "_position.npy"), allow_pickle=True)

