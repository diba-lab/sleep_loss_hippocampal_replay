import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fileUtilities as fd


basepath = '~/Documents/ClusteringHub/RatJ_2019-05-31_03-55-36/'


dir_inside = fd.listDirectory(basepath, 'Shank')


# dir_path = os.path.dirname(workDir1)
# dir_in = os.listdir(os.path.expanduser(workDir1))
# output = [dI for dI in os.listdir(
#     workDir1) if os.path.isdir(os.path.join(workDir1, dI))]
allspikes = []
for idx, fd in enumerate(dir_inside):
    filename = basepath + fd + '/' + fd + '.csv'
    quality_filename = basepath + fd + '/' + fd + '_quality.csv'

    spkInfo = pd.read_csv(filename, header=None, names=[
        'spktimes', 'ClusterNum', 'MaxSite'])

    quality_Info = pd.read_csv(quality_filename, header=None)
    clustertype = quality_Info[-1].tolist()

    spktimes = spkInfo['spktimes']
    allspikes.append(spktimes.tolist())

allspikes = [item for sublist in allspikes for item in sublist]
histspk, edges = np.histogram(allspikes, 14)

plt.plot(histspk)
