import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# import scipy as sc

basefolder = "/data/Clustering/SleepDeprivation/RatN/Day1/"
eventsfolder = basefolder + "events/Epochs/"
Epochs = eventsfolder + "text.npy"
fileTime = eventsfolder + "timestamps.npy"
timestamps = np.load(fileTime)
time_text = np.load(Epochs)

fileName = basefolder + "RatN_Day1_2019-10-09_03-52-32.eeg"
nChansEEG = 134
SampFreq = 1250
Data = np.memmap(fileName, dtype="int16", mode="r")
total_time = len(Data) / (nChansEEG * 1250)

# Defining time windows
timestamps = timestamps / 30000
pre_time = np.array([0, timestamps[0]])
maze_time = np.array([timestamps[1], timestamps[2]])
post_time = np.array([timestamps[3], total_time])
sleepdep_time = np.array([timestamps[3], timestamps[4]])
duration = 191 * 60  # in seconds
noisy_epoch = [191 * 60, 192 * 60]
noisy_duration = np.diff(noisy_epoch)

epoch_times = {
    "PRE": pre_time,
    "MAZE": maze_time,
    "POST": post_time,
    "SD": sleepdep_time,
}


for i in range(len(noisy_epoch)):
    for key, value in epoch_times.items():
        if epoch_times[key][0] > noisy_epoch[0]:
            epoch_times[key][0] = epoch_times[key][0] - noisy_duration
        if epoch_times[key][1] > noisy_epoch[0]:
            epoch_times[key][1] = epoch_times[key][1] - noisy_duration


np.save(basefolder + "epochs.npy", epoch_times)

dd = np.load(basefolder + "epochs.npy", allow_pickle=True)
