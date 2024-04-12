import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as sg
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

sampleRate = 30000
basePath = Path(
    "/home/bapung/Documents/ClusteringHub/spykcirc/RatK/Day1/RatK_Day1_2019-08-06_03-44-01-1.GUI"
)


clustergrp = pd.read_csv(basePath / "cluster_group.tsv", sep="\t")
clusterinfo = pd.read_csv(basePath / "cluster_info.tsv", sep="\t")
clusterpurity = pd.read_csv(basePath / "cluster_purity.tsv", sep="\t")
ampscale = np.load(basePath / "amplitudes.npy")
pcfeatures = np.load(basePath / "pc_features.npy")
spiketemplates = np.load(basePath / "spike_templates.npy")
spiketimes = np.load(basePath / "spike_times.npy")
# templatefeatures = np.load(basePath / "template_features.npy")

noise_clusters = clustergrp[clustergrp["group"] == "noise"].cluster_id.values

rows_clugrp = [
    ind
    for ind, id_ in enumerate(clustergrp["cluster_id"].values)
    if id_ in noise_clusters
]
rows_cluinfo = [
    ind for ind, id_ in enumerate(clusterinfo["id"].values) if id_ in noise_clusters
]
rows_cluprity = [
    ind
    for ind, id_ in enumerate(clusterpurity["cluster_id"].values)
    if id_ in noise_clusters
]

clustergrp = clustergrp.drop(rows_clugrp)
clusterinfo = clusterinfo.drop(rows_cluinfo)
clusterpurity = clusterpurity.drop(rows_cluprity)


cluind2delete = []
for clu in noise_clusters:
    cluind2delete.extend(np.where(spiketemplates == clu)[0])
cluind2delete = np.asarray(cluind2delete)

ampscale = np.delete(ampscale, cluind2delete, axis=0)
pcfeatures = np.delete(pcfeatures, cluind2delete, axis=0)
spiketemplates = np.delete(spiketemplates, cluind2delete, axis=0)
spiketimes = np.delete(spiketimes, cluind2delete, axis=0)


deadfile = Path(
    "/data/Clustering/SleepDeprivation/RatK/Day1/RatK_Day1_2019-08-06_03-44-01.dead"
)
with deadfile.open("r") as f:
    noisy = []
    for line in f:
        epc = line.split(" ")
        epc = [int((float(_) / 1000) * sampleRate) for _ in epc]
        noisy.append(epc)

ind2delete = []
for (start, stop) in noisy:
    ind2delete.extend(np.where((spiketimes > start) & (spiketimes < stop))[0])

ind2delete = np.asarray(ind2delete)


# ==== which clu has these spikes======

clu_has = np.unique(spiketemplates[ind2delete])
for clu in clu_has:

    noisy_temp = len(np.where(spiketemplates == clu)[0])
    n_noisy_clu = len(np.where(spiketemplates[ind2delete] == clu)[0])
    if noisy_temp == n_noisy_clu:
        clustergrp = clustergrp.drop(
            np.where(clusterinfo["cluster_id"].values == clu)[0]
        )
        clusterinfo = clusterinfo.drop(np.where(clusterinfo["id"].values == clu)[0])
        clusterpurity = clusterpurity.drop(
            np.where(clusterinfo["cluster_id"].values == clu)[0]
        )


ampscale = np.delete(ampscale, ind2delete, axis=0)
pcfeatures = np.delete(pcfeatures, ind2delete, axis=0)
spiketemplates = np.delete(spiketemplates, ind2delete, axis=0)
spiketimes = np.delete(spiketimes, ind2delete, axis=0)

# for clu in clusterinfo["id"].values:
#     nspikes = np.where([nspikes])

np.save(basePath / "amplitudes.npy", ampscale)
np.save(basePath / "pc_features.npy", pcfeatures)
np.save(basePath / "spike_templates.npy", spiketemplates)
np.save(basePath / "spike_times.npy", spiketimes)

clustergrp.to_csv(basePath / "cluster_group.tsv", sep="\t")
clusterinfo.to_csv(basePath / "cluster_info.tsv", sep="\t")
clusterpurity.to_csv(basePath / "cluster_purity.tsv", sep="\t")
