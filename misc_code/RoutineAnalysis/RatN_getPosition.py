import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

# check_file = pd.read_csv(fileName, header=5)
# firstLine = pd.read_csv(fileName, nrows=0).iloc[:, 3]


class ExtractPosition:

    nChans = 134
    sRate = 30000
    binSize = 0.250  # in seconds
    timeWindow = 3600  # Number of bins (15 minutes)

    def __init__(self, basePath):
        # self.sessionName = os.path.basename(os.path.normpath(basePath))
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath

        posFolder = basePath + "position/"
        for file in os.listdir(posFolder):
            if file.endswith(".csv"):
                print(file)
                fileName = posFolder + file

        with open(fileName, newline="") as f:
            reader = csv.reader(f)
            row1 = next(reader)
            StartTime = [
                row1[i + 1] for i in range(len(row1)) if row1[i] == "Capture Start Time"
            ]

        positionStruct = pd.read_csv(fileName, header=5)
        # TODO get automatic column location
        positionStruct = positionStruct.iloc[:, [1, 6, 7, 8]]
        positionStruct.interpolate(axis=0)

        self.time = positionStruct.iloc[:, 0]
        self.posX = positionStruct.iloc[:, 1]
        self.posY = positionStruct.iloc[:, 2]
        self.posZ = positionStruct.iloc[:, 3]
        self.dt = self.time[1] - self.time[0]

        self.time = self.time + 5.457 + 10041.21
        posVar = {}
        posVar["X"] = self.posX
        posVar["Y"] = self.posZ
        posVar["time"] = self.time
        # posVar["Speed"] = Speed(self)

        np.save(basePath + "position.npy", posVar)

    def plotPosition(self):

        plt.clf()
        plt.plot(self.posX, self.posZ, ".")

    def Speed(self):
        location = np.sqrt((self.posZ) ** 2 + (self.posX) ** 2)
        spd = np.abs(np.diff(location)) / self.dt

        self.speed = spd.tolist()
        return self.speed


basePath = "/data/Clustering/SleepDeprivation/RatN/Day2/"

RatNDay2 = ExtractPosition(basePath)
# velocity = RatNDay2.Speed()
# plt.clf()
# plt.plot(velocity)
