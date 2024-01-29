import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import datetime as dt

# import scipy as sc

basefolder = "/data/Clustering/SleepDeprivation/RatN/Day2/"


dat1 = basefolder + "RatN_Day2_2019-10-11_03-58-54_part1.dat"
pre_dat = np.memmap(dat1, mode="r", dtype="int16")

time_pre = len(pre_dat) / (134 * 30000)


file = "/data/Clustering/SleepDeprivation/RatN/Day2/2019-10-11_07-18-56/experiment1/recording1/continuous/Rhythm_FPGA-100.0/timestamps.npy"

timestamp_afterpre_file = np.load(file)
timestamp_afterpre = len(np.load(file)) / 30000


eventsfolder1 = (
    basefolder
    + "2019-10-11_03-58-54/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/"
)
eventsfolder2 = (
    basefolder
    + "2019-10-11_03-58-54/experiment2/recording1/events/Message_Center-904.0/TEXT_group_1/"
)
eventsfolder3 = (
    basefolder
    + "2019-10-11_07-18-56/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/"
)
Epochs1 = eventsfolder1 + "text.npy"
Epochs2 = eventsfolder2 + "text.npy"
Epochs3 = eventsfolder3 + "text.npy"

fileTime1 = eventsfolder1 + "timestamps.npy"
fileTime2 = eventsfolder2 + "timestamps.npy"
fileTime3 = eventsfolder3 + "timestamps.npy"

timestamps1 = np.load(fileTime1)
time_text1 = np.load(Epochs1)

timestamps2 = np.load(fileTime2)
time_text2 = np.load(Epochs2)

timestamps3 = np.load(fileTime3)
time_text3 = np.load(Epochs3)


# in seconds (start recording (07-18-56) - start tracking)(07.19.01.457)
maze_start = 5.457
maze_duration = (timestamps3[0] - timestamp_afterpre_file[0]) / (30000)

diff_post_maze = (timestamps3[1] - timestamps3[0]) / (30000 * 60)


pre_time = np.array([0, time_pre])
maze_time = np.array([time_pre + maze_start, time_pre + maze_start + maze_duration])
post_time = np.array(
    [
        time_pre + maze_start + maze_duration + diff_post_maze,
        timestamp_afterpre + time_pre,
    ]
)


# duration = 168 * 60  # in seconds
noisy_epoch = [168 * 60, 172 * 60]
noisy_duration = np.diff(noisy_epoch)

epoch_times = {"PRE": pre_time, "MAZE": maze_time, "POST": post_time}


for i in range(len(noisy_epoch)):
    for key, value in epoch_times.items():
        if epoch_times[key][0] > noisy_epoch[0]:
            epoch_times[key][0] = epoch_times[key][0] - noisy_duration
        if epoch_times[key][1] > noisy_epoch[0]:
            epoch_times[key][1] = epoch_times[key][1] - noisy_duration


np.save(basefolder + "epochs.npy", epoch_times)

# dd = np.load(basefolder + "epochs.npy", allow_pickle=True)
