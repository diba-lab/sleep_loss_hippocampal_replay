import numpy as np
import matplotlib.pyplot as plt

# Name of source file
filename = "source_file.dat"

# Name of destination file
Destfile = "destination_file.dat"

nChans = 134  # number of channels in your dat file
SampFreq = 30000

# start and end time which you want to extract
start_time = 20  # from this time in seconds
end_time = 60 * 60  # duration of chunk

# read required chunk from the source file
b1 = np.memmap(
    filename,
    dtype="int16",
    offset=2 * nChans * SampFreq * start_time,
    mode="r",
    shape=(nChans * SampFreq * end_time),
)

# allocates space for the file
c = np.memmap(Destfile, dtype="int16", mode="w+", shape=(len(b1)))
c[: len(b1)] = b1
# del c

# writes the data to that space
d = np.memmap(Destfile, dtype="int16", mode="r+", shape=(len(b1)))
