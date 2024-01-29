import sys
import matplotlib.pyplot as plt
from phylib.io.model import load_model

# from phylib.utils.color import selected_cluster_color

folderPath = "/data/Clustering/SleepDeprivation/RatN/Day1/RatN__2019-10-09_03-52-32/experiment1/recording1/continuous/Rhythm_FPGA-100.0/Shank8/"
# First, we load the TemplateModel.
model = load_model(sys.argv[1])  # first argument: path to params.py

# We obtain the cluster id from the command-line arguments.
cluster_id = int(sys.argv[2])  # second argument: cluster index

# We get the waveforms of the cluster.
waveforms = model.get_cluster_spike_waveforms(cluster_id)
n_spikes, n_samples, n_channels_loc = waveforms.shape

# We get the channel ids where the waveforms are located.
channel_ids = model.get_cluster_channels(cluster_id)

# We plot the waveforms on the first four channels.
f, axes = plt.subplots(1, min(4, n_channels_loc), sharey=True)
for ch in range(min(4, n_channels_loc)):
    axes[ch].plot(waveforms[::100, :, ch].T, c=selected_cluster_color(0, 0.05))
    axes[ch].set_title("channel %d" % channel_ids[ch])
plt.show()
