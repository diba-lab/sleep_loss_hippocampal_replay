import numpy as np
import os
from pathlib import Path as pth
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from spikesUtil import ExtractSpikes as getspk


basePath = "/data/Clustering/SleepDeprivation/RatJ/Day1/"
spklist = []
for i in range(2, 9):

    clupath = pth(basePath, "Shank" + str(i), "RatJDay1_Shank" + str(i) + ".clu.1")
    clu = []
    with open(clupath) as f:

        for line in f:
            clu.append(int(line))

    num_clust = clu[0]
    clust_id = np.unique(clu[1:])
    spk = np.asarray(clu[1:])

    respath = pth(basePath, "Shank" + str(i), "RatJDay1_Shank" + str(i) + ".res.1")
    res = []
    with open(respath) as f:

        for line in f:
            res.append(int(line))

    spk_time = np.asarray(res)

    filename = pth(basePath, "Shank" + str(i), "RatJDay1_Shank" + str(i) + ".xml")
    myroot = ET.parse(filename).getroot()

    chan_session = []
    clus_pyr = []

    for x in myroot.findall("units"):
        for y in x.findall("unit"):
            for z, z1 in zip(y.findall("quality"), y.findall("cluster")):

                if z.text is not None:
                    if (z.text).isdigit():

                        clus_pyr.append([int(z.text), int(z1.text)])

    clus_pyr = np.asarray(clus_pyr)
    pyr_id = clus_pyr[clus_pyr[:, 0] == 9, 1]

    for clus in pyr_id:
        spklist.append(spk_time[np.where(spk == clus)[0]] / 30000)
        # spklist.append([i for i in range(len(spk)) if spk[i] == clus])


# for ind, cell in enumerate(spklist):
#     plt.plot(cell, ind * np.ones(len(cell)), ".")

np.save(basePath + "RatJ_Day1_2019-05-31_03-55-36" + "_spikes.npy", spklist)


# sess1 = getspk(basePath)
# sess1.CollectSpikes()
# sess1.ExpVAr()
#     m = "".join(line)
# class ExtractfromClu:
#     def __init__(self, basePath):
#         # self.sessionName = os.path.basename(os.path.normpath(basePath))
#         self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
#         self.basePath = basePath

#         for file in os.listdir(basePath):
#             if file.endswith(".eeg"):

#                 self.subname = file[:-4]
#                 self.filename = os.path.join(basePath, file)
#                 self.filePrefix = os.path.join(basePath, file[:-4])

#     def clu2Spike(self):
#         filepath = pth(self.basePath, "Shank4", "RatJDay1_Shank2.clu.1")
#         with open(filepath) as f:
#             next(f)
#             for i, line in enumerate(f):

#                 m = "".join(line)

#                 if "KeyCount" in m:
#                     track_begin = i + 2
#                     line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
#                     total_frames = int(line_frame[1]) - 1
#                     break


# basePath = [
#     "/data/Clustering/SleepDeprivation/RatJ/Day1/",
#     # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
#     # "/data/Clustering/SleepDeprivation/RatK/Day1/",
#     # "/data/Clustering/SleepDeprivation/RatK/Day2/",
#     # "/data/Clustering/SleepDeprivation/RatN/Day1/",
#     # "/data/Clustering/SleepDeprivation/RatN/Day2/",
# ]

# with open(fileName) as f:
#     next(f)
#     for i, line in enumerate(f):

#         m = "".join(line)

#         if "KeyCount" in m:
#             track_begin = i + 2
#             line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
#             total_frames = int(line_frame[1]) - 1
#             break


# spkgen = [ExtractfromClu(x) for x in basePath]

# for i, sess in enumerate(spkgen):
#     sess.clu2Spike()

