import numpy as np
from scipy import stats
import scipy.signal as sg
import pingouin as pg
import pandas as pd
from neuropy import core
from scipy import interpolate


def get_frate_in_epochs(
    neurons: core.Neurons,
    epochs: core.Epoch,
    brainstates: core.Epoch,
    artifact: core.Epoch,
    as_df=True,
):
    frate_epochs = []
    for e in epochs.itertuples():
        e_states = brainstates.time_slice(e.start, e.stop, strict=False)
        if artifact is not None:
            e_artifact = artifact.time_slice(e.start, e.stop, strict=False)
            # e_artifact_duration = e_artifact.durations.sum()  # Bug?
            e_artifact_duration = core.epoch.get_epoch_overlap_duration(e_states, e_artifact)
        else:
            e_artifact_duration = 0

        if e_states.n_epochs > 0:
            e_duration = e_states.durations.sum()
            e_neurons = neurons.get_neurons_in_epochs(e_states)
            e_nspikes = e_neurons.n_spikes
            frate_ = e_nspikes / (e_duration - e_artifact_duration)
        else:
            frate_ = np.nan * np.zeros(neurons.n_neurons)

        frate_epochs.append(frate_)

    frate_epochs = np.asarray(frate_epochs).T

    if as_df:
        df = pd.DataFrame(frate_epochs, columns=epochs.labels)
        df["neuron_id"] = neurons.neuron_ids
        df["neuron_type"] = neurons.neuron_type
        return df
    else:
        return frate_epochs