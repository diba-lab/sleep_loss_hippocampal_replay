# cell variability


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
from phylib.stats import correlograms, firing_rate

basePath = "/data/Clustering/SleepDeprivation/RatN/Day2/"

# RatNDay2 = PlaceField(folderPath)

# RatNDay2.pfPlot()

# self.sessionnName = os.path.basename(os.path.normpath(basePath))
ssessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
basePath = basePath
for file in os.listdir(basePath):
    if file.endswith(".eeg"):
        print(file)
        subname = file[:-4]
        print(os.path.join(basePath, file))

spikes = np.load(basePath + subname + "_spikes.npy", allow_pickle=True)
position = np.load(basePath + subname + "_position.npy", allow_pickle=True)
epochs = np.load(basePath + "epochs.npy", allow_pickle=True)


pre = epochs.item().get("PRE")
post = epochs.item().get("POST")

# xcoord = position.item().get("X")
# ycoord = position.item().get("Y")

# diff_posx = np.diff(xcoord)
# diff_posy = np.diff(ycoord)
# timepos = position.item().get("time")

# dt = timepos[1] - timepos[0]

# location = np.sqrt((xcoord) ** 2 + (ycoord) ** 2)
# speed = np.abs(np.diff(location)) / dt

# spdt = timepos

spktime = spikes

# xmesh = np.arange(min(xcoord), max(xcoord) + 1, 2)
# ymesh = np.arange(min(ycoord), max(ycoord) + 1, 2)
# xx, yy = np.meshgrid(xmesh, ymesh)
# pf2, xe1, ye1 = np.histogram2d(xcoord, ycoord, bins=[xmesh, ymesh])


binsize = 500  # in seconds
nPyr = 10
spk_variablity = []
f_rate = []
# for cell in range(len(spktime)):

#     spkt = spktime[cell]
#     recording_dur = post[1] - pre[0]
#     spkbin = np.arange(pre[0], post[1], binsize)
#     spkcount, _ = np.histogram(spkt, bins=spkbin)
#     frac_change = np.diff(spkcount) / spkcount[:-1]
#     spk_variablity.append(np.nanstd(frac_change))
#     f_rate.append(len(spkt) / recording_dur)

# plt.clf()
# plt.plot(f_rate, spk_variablity, ".")

b = spktime[1].reshape((len(spktime[1]),))

a = (b * 30000).astype(np.int64)

ccg1 = correlograms(
    b,
    np.ones(len(b)),
    cluster_ids=[1],
    sample_rate=30000,
    bin_size=0.001,
    window_size=0.1,
)


# for i in [1]:
#     for j in [2]:
#         spkt1 = spktime[i]
#         spkt2 = spktime[j]

#         bn_refrence = []

#         spk_diff = spkt1 - spkt2.T
