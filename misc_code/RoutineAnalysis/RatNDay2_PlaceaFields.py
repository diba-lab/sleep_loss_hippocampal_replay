import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

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


xcoord = position.item().get("X")
ycoord = position.item().get("Y")

diff_posx = np.diff(xcoord)
diff_posy = np.diff(ycoord)
timepos = position.item().get("time")

dt = timepos[1] - timepos[0]

location = np.sqrt((xcoord) ** 2 + (ycoord) ** 2)
speed = np.abs(np.diff(location)) / dt

spdt = timepos

spktime = spikes

xmesh = np.arange(min(xcoord), max(xcoord) + 1, 2)
ymesh = np.arange(min(ycoord), max(ycoord) + 1, 2)
xx, yy = np.meshgrid(xmesh, ymesh)
pf2, xe1, ye1 = np.histogram2d(xcoord, ycoord, bins=[xmesh, ymesh])

plt.clf()
nPyr = 10
for cell in range(10):
    spkt = spktime[cell]
    spd_spk = np.interp(spkt, spdt[:-1], speed)

    spkt = spkt[spd_spk > 50]  # only selecting spikes where rat's speed is  > 5 cm/s

    spktx = np.interp(spkt, timepos, xcoord)
    spkty = np.interp(spkt, timepos, ycoord)

    spktx = spktx.reshape((len(spktx)))
    spkty = spkty.reshape((len(spkty)))

    pf, xe, ye = np.histogram2d(spktx, spkty, bins=[xmesh, ymesh])

    pft = pf2 * (1 / 30)

    eps = np.spacing(1)
    pfRate = pf / (pft + eps)

    pfRate_smooth = gaussian_filter(pfRate, sigma=3)

    #    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))

    nRows = np.ceil(np.sqrt(nPyr))
    nCols = np.ceil(np.sqrt(nPyr))

    #    plt.plot(posx_mz,posy_mz,'.')
    plt.subplot(nRows, nCols, cell + 1)
    plt.imshow(pfRate_smooth)


# folderPath = "/data/Clustering/SleepDeprivation/RatN/Day1/"

# RatNDay1 = ExtractSpikes(folderPath)
# RatNDay1.CollectSpikes()
# RatNDay1.ExpVAr()


# class PlaceField:

#     nChans = 16
#     sRate = 30000
#     binSize = 0.250  # in seconds
#     timeWindow = 3600  # in seconds

#     def __init__(self, basePath):
#         # self.sessionnName = os.path.basename(os.path.normpath(basePath))
#         self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
#         self.basePath = basePath
#         for file in os.listdir(basePath):
#             if file.endswith(".eeg"):
#                 print(file)
#                 self.subname = file[:-4]
#                 print(os.path.join(basePath, file))

#         self.spikes = np.load(
#             self.basePath + self.subname + "_spikes.npy", allow_pickle=True
#         )

#         self.position = np.load(
#             self.basePath + self.subname + "_position.npy", allow_pickle=True
#         )

#     def pfPlot(self):

#         xcoord = self.position.item().get("X")
#         ycoord = self.position.item().get("Y")
#         timepos = self.position.item().get("time")

#         dt = timepos[1] - timepos[0]

#         location = np.sqrt((xcoord) ** 2 + (ycoord) ** 2)
#         speed = np.abs(np.diff(location)) / dt

#         spdt = timepos

#         spktime = self.spikes

#         xmesh = np.arange(min(xcoord), max(xcoord) + 1, 2)
#         ymesh = np.arange(min(ycoord), max(ycoord) + 1, 2)
#         xx, yy = np.meshgrid(xmesh, ymesh)
#         pf2, xe1, ye1 = np.histogram2d(xcoord, ycoord, bins=[xmesh, ymesh])

#         nPyr = 10
#         for cell in range(10):
#             spkt = spktime[cell]
#             # spd_spk = np.interp(spkt, spdt, speed)

#             # spkt = spkt[
#             #     spd_spk > 5
#             # ]  # only selecting spikes where rat's speed is  > 5 cm/s

#             spktx = np.interp(spkt, timepos, xcoord)
#             spkty = np.interp(spkt, timepos, ycoord)

#             pf, xe, ye = np.histogram2d(spktx, spkty, bins=[xmesh, ymesh])

#             pft = pf2 * (1 / 30)

#             eps = np.spacing(1)
#             pfRate = pf / (pft + eps)

#             pfRate_smooth = gaussian_filter(pfRate, sigma=3)

#             #    ICA_strength = np.array(np.transpose(fICAStrength['ActStrength']['subjects'][sub_name]['wake'][:]))

#             nRows = np.ceil(np.sqrt(nPyr))
#             nCols = np.ceil(np.sqrt(nPyr))

#             #    plt.plot(posx_mz,posy_mz,'.')
#             plt.subplot(nRows, nCols, cell + 1)
#             plt.imshow(pfRate_smooth)


# # folderPath = "/data/Clustering/SleepDeprivation/RatN/Day1/"

# # RatNDay1 = ExtractSpikes(folderPath)
# # RatNDay1.CollectSpikes()
# # RatNDay1.ExpVAr()


# folderPath = "/data/Clustering/SleepDeprivation/RatN/Day2/"

# RatNDay2 = PlaceField(folderPath)

# RatNDay2.pfPlot()
