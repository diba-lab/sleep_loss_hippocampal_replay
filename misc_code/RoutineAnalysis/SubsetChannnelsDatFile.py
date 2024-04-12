import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# TODO extract shanks
folderPath = "/data/Clustering/SleepDeprivation/RatK/Day1/2019-08-06_03-44-01/experiment1/recording1/continuous/Rhythm_FPGA-100.0/"

SourceFile = "continuous.dat"
# cmd = "wc -l my_text_file.txt > out_file.txt"
# os.system(cmd)


# DestFolder = Shank2

# cmnd = (
#     "process_extractchannels "
#     + folderPath
#     + SourceFile
#     + try_subChan.dat
#     + "134 45 34 46 33 47 32 44 35 43 36 42 37 41 38 40 39"
# )

# print(cmnd)
Shank2map = "29 18 30 17 31 16 28 19 27 20 26 21 22 25 24 23"
Shank4map = "61 50 62 49 63 48 60 51 59 52 58 53 57 54 56 55"

p = subprocess.Popen(
    [
        "process_extractchannels",
        "continuous.dat Shank2.dat 134 29 18 30 17 31 16 28 19 27 20 26 21 22 25 24 23",
    ],
    cwd=folderPath,
)

p.wait()

p = subprocess.Popen(
    [
        "process_extractchannels",
        "continuous.dat",
        "Shank4.dat",
        "134",
        "61 50 62 49 63 48 60 51 59 52 58 53 57 54 56 55",
    ],
    cwd=folderPath,
)

p.wait()
