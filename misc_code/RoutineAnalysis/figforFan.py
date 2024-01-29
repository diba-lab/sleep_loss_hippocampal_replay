import numpy as np
import matplotlib.pyplot as plt
import os


basePath = "/data/Clustering/SleepDeprivation/RatK/Day1/"

for file in os.listdir(basePath):
    if file.endswith(".eeg"):
        subname = file[:-4]
        filename = os.path.join(basePath, file)
        filePrefix = os.path.join(basePath, file[:-4])

nchans = 134
srate = 30000
ripples = np.load(filePrefix + "_ripples.npy", allow_pickle=True)
basics = np.load(filePrefix + "_basics.npy", allow_pickle=True)
chanmap = basics.item().get("channels")

ripple_time = ripples.item().get("timestamps")
ripple_ind = 700
ripple1 = int(ripple_time[ripple_ind][0] * (30000 / 1250))
dur_ripple = int(np.diff(ripple_time[ripple_ind]) * 30000 / 1250)
data = np.memmap(
    filePrefix + ".dat",
    offset=(ripple1 - 10000) * nchans * 2,
    dtype="int16",
    mode="r",
    shape=(nchans * srate * 1),
)

nframes = int(len(data) / 134)
data_ripp = np.reshape(data, (nframes, nchans))


t = np.arange(-10000 / srate, 20000 / srate, 1 / srate)


plt.clf()
for i, chan in enumerate(chanmap[16:28]):
    chandata = data_ripp[:, chan]
    strt_ind = np.arange(10000, 11000 + dur_ripple)
    plt.plot(t, chandata - i * 2000, color="gray", linewidth=1)
    # plt.plot(t[strt_ind], chandata[strt_ind] - i * 2000, color="orange")

# plt.Line2D([t[10000], t[10000]], [-3000, 0])

plt.axis("off")

