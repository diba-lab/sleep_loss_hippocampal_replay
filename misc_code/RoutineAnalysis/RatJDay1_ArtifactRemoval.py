import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

folderPath = "/data/Clustering/SleepDeprivation/RatJ/Day1/og_files/"
fileName = folderPath + "RatJ_2019-05-31_03-55-36.eeg"


nChans = 75
SampFreq = 1250


Data = np.memmap(fileName, dtype="int16", mode="r")
# lenData = len(Data) / 75
# np.save(folderPath + "numframes_OG.npy", lenData)
Data = np.memmap.reshape(Data, (int(len(Data) / 75), 75))

chanData = Data[:, 17]

zsc = np.abs(stat.zscore(chanData))

artifact_binary = np.where(zsc > 1.5, 1, 0)
artifact_binary = np.concatenate(([0], artifact_binary, [0]))
artifact_diff = np.diff(artifact_binary)


artifact_start = np.where(artifact_diff == 1)[0]
artifact_end = np.where(artifact_diff == -1)[0]

firstPass = np.vstack((artifact_start - 2, artifact_end + 2)).T


minInterArtifactDist = 5 * SampFreq
secondPass = []
artifact = firstPass[0]
for i in range(1, len(artifact_start)):
    if firstPass[i, 0] - artifact[1] < minInterArtifactDist:
        # Merging artifacts
        artifact = [artifact[0], firstPass[i, 1]]
    else:
        secondPass.append(artifact)
        artifact = firstPass[i]

secondPass.append(artifact)
secondPass = np.asarray(secondPass)

np.save(folderPath + "noisy_timestamps_fromOG.npy", secondPass)

Data_start = np.concatenate(([0], secondPass[:, 1])) / 1250
DatFileOG = folderPath + "/RatJ_2019-05-31_03-55-36.dat"

endFrametime = len(np.memmap(DatFileOG, dtype="int16", mode="r")) / (75 * 30000)
Data_end = np.concatenate((secondPass[:, 0], [len(zsc)])) / 1250


DestFolder = folderPath + "/RatJ_2019-05-31_03-55-36_denoised.dat"

nChans = 75
SampFreq = 30000

b = []
for i in range(len(Data_start) - 1):
    print(i)
    start_time = Data_start[i]
    end_time = Data_end[i]

    duration = end_time - start_time  # in seconds
    b.append(
        np.memmap(
            DatFileOG,
            dtype="int16",
            mode="r",
            offset=2 * nChans * int(SampFreq * start_time),
            shape=(nChans * int(SampFreq * duration)),
        )
    )

c = np.memmap(DestFolder, dtype="int16", mode="w+", shape=sum([len(x) for x in b]))

del c
d = np.memmap(DestFolder, dtype="int16", mode="r+", shape=sum([len(x) for x in b]))

sizeb = [0]
sizeb.extend([len(x) for x in b])
sizeb = np.cumsum(sizeb)

for j in range(len(b)):

    d[sizeb[j] : sizeb[j + 1]] = b[j]
    # d[len(b[i]) : len(b1) + len(b2)] = b2
del d
del b
