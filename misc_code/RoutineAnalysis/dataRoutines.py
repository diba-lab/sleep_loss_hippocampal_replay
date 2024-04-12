import numpy as np
import os
from MakePrmKlusta import makePrmPrb
from makeChanMap import ExtractChanXml
from pathlib import Path


class firstRun:
    shanks_all = 8

    def __init__(self, basePath):
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.basePath = basePath
        for file in os.listdir(basePath):
            if file.endswith(".eeg"):
                self.subname = file[:-4]
                self.filename = os.path.join(basePath, file)
                self.filePrefix = os.path.join(basePath, file[:-4])

    def genBasics(self):
        #%% generate basics such as channels, sampling rate

        ExtractChanXml(self.basePath)

    def genKlusta(self):
        #%% generates files for klusta

        prmGen = [
            makePrmPrb(basePath[i], shanks_all[i], shanks_Chan_all[i])
            for i in range(nSessions)
        ]

        for i in range(nSessions):
            prmGen[i].makePrm()
            prmGen[i].makePrb()
            prmGen[i].makePrmServer()
            prmGen[i].makePrbServer()

    def genPosition(self):
        #%% generates best ripple channel
        pass

    def genTheta(self):
        #%% generates best theta channel

        pass

    def genRipple(self):
        #%% generates best ripple channel
        pass

    def getSpikes(self):
        #%% generates best ripple channel
        pass


basePath = [
    # "/data/Clustering/SleepDeprivation/RatJ/Day1/",
    # "/data/Clustering/SleepDeprivation/RatJ/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day1/",
    # "/data/Clustering/SleepDeprivation/RatK/Day2/",
    # "/data/Clustering/SleepDeprivation/RatK/Day3/",
    # "/data/Clustering/SleepDeprivation/RatK/Day4/",
    # "/data/Clustering/SleepDeprivation/RatN/Day1/",
    "/data/Clustering/SleepDeprivation/RatN/Day2/",
    # "/data/Clustering/SleepDeprivation/RatN/Day3/",
    # "/data/Clustering/SleepDeprivation/RatN/Day4/",
]
nSessions = len(basePath)
shanks_all = [8]
shanks_Chan_all = [8]

