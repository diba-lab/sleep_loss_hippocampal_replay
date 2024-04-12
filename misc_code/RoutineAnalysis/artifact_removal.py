# removing high amplitude artifact from data

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

folderPath = "/data/Clustering/SleepDeprivation/RatK/Day2/"
fileName = folderPath + "RatK-Day2-2019-08-08_04-00-00.eeg"

fileOGDat = folderPath + "RatK-Day2-2019-08-08_04-00-00.dat"


class ArtifactRemove:
    nChans = 134
    SampFreq = 1250
    ArtifactThresh = 1.5
    MinEpochSize = 5  # miniumum duration between two artifact periods

    def __init__(self, basepath, goodChan):
        self.basepath = basepath
        self.goodChan = goodChan

    def findArtifactEpochs(self):
        Data = np.memmap(fileName, dtype="int16", mode="r")
        Data2 = np.memmap(fileOGDat, dtype="int16", mode="r")

        Data1 = np.memmap.reshape(Data, (int(len(Data) / nChansEEG), nChansEEG))

        chanData = Data1[:, self.goodChan]

        zsc = np.abs(stat.zscore(chanData))

        artifact_binary = np.where(zsc > self.ArtifactThresh, 1, 0)
        artifact_binary = np.concatenate(([0], artifact_binary, [0]))
        artifact_diff = np.diff(artifact_binary)

        artifact_start = np.where(artifact_diff == 1)[0]
        artifact_end = np.where(artifact_diff == -1)[0]

        # making outer bound for the periods of artifact
        firstPass = np.vstack((artifact_start - 2, artifact_end + 2)).T

        # merging short close artifact periods
        minInterArtifactDist = self.MinEpochSize * self.SampFreq
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

        Data_start = np.concatenate(([0], secondPass[:, 1])) / 1250

        DataShank = folderPath + "Shank4/RatKDay2Shank4.dat"
        endFrametime = len(np.memmap(DataShank, dtype="int16", mode="r")) / (16 * 30000)
        Data_end = np.concatenate((secondPass[:, 0] / 1250, [endFrametime]))

    def RemoveNoise(self):
        for shankID in range(4, 9):
            print(shankID)

            DatFileOG = (
                folderPath
                + "Shank"
                + str(shankID)
                + "/RatKDay2Shank"
                + str(shankID)
                + ".dat"
            )
            DestFolder = (
                folderPath
                + "Shank"
                + str(shankID)
                + "/RatKDay2Shank"
                + str(shankID)
                + "_denoised.dat"
            )

            nChans = 16
            SampFreq = 30000

            b = []
            for i in range(len(Data_start)):

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

            c = np.memmap(
                DestFolder, dtype="int16", mode="w+", shape=sum([len(x) for x in b])
            )

            del c
            d = np.memmap(
                DestFolder, dtype="int16", mode="r+", shape=sum([len(x) for x in b])
            )

            sizeb = [0]
            sizeb.extend([len(x) for x in b])
            sizeb = np.cumsum(sizeb)

            for i in range(len(b)):

                d[sizeb[i] : sizeb[i + 1]] = b[i]
                # d[len(b[i]) : len(b1) + len(b2)] = b2
            del d
            del b


nChansEEG = 134
SampFreq = 1250


Data = np.memmap(fileName, dtype="int16", mode="r")
Data2 = np.memmap(fileOGDat, dtype="int16", mode="r")


Data1 = np.memmap.reshape(Data, (int(len(Data) / nChansEEG), nChansEEG))

chanData = Data1[:, 17]

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

Data_start = np.concatenate(([0], secondPass[:, 1])) / 1250

DataShank = folderPath + "Shank4/RatKDay2Shank4.dat"
endFrametime = len(np.memmap(DataShank, dtype="int16", mode="r")) / (16 * 30000)
Data_end = np.concatenate((secondPass[:, 0] / 1250, [endFrametime]))


for shankID in range(4, 9):
    print(shankID)

    DatFileOG = (
        folderPath + "Shank" + str(shankID) + "/RatKDay2Shank" + str(shankID) + ".dat"
    )
    DestFolder = (
        folderPath
        + "Shank"
        + str(shankID)
        + "/RatKDay2Shank"
        + str(shankID)
        + "_denoised.dat"
    )

    nChans = 16
    SampFreq = 30000

    b = []
    for i in range(len(Data_start)):

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

    for i in range(len(b)):

        d[sizeb[i] : sizeb[i + 1]] = b[i]
        # d[len(b[i]) : len(b1) + len(b2)] = b2
    del d
    del b
