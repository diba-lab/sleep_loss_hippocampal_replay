import numpy as np
from neuropy import core
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats


def ripple_modulation(neurons: core.Neurons, starts, peaks, stops, n_bins=4):
    """Neurons firing rate modulation within ripples
    Each ripple is divided into following sub-epochs of size n_bins:
        before-start, start-peak, peak-stop, stop-post

    Parameters
    ----------
    neurons : Neurons
        [description]
    starts : array
        ripple start times in seconds
    peaks : array
        ripple peak times in seconds
    stops : array
        ripple
    nbins : int, optional
        number of bins, by default 4

    Returns
    -------
    array: n_neurons x (4*n_bins)

    References
    ----------
    1. Diba et al. 2014
    2. Cscisvari et al. 1999
    """

    start_peak_dur, peak_stop_dur = peaks - starts, stops - peaks
    pre_start = core.Epoch.from_array(starts - start_peak_dur, starts)
    start_peak = core.Epoch.from_array(starts, peaks)
    peak_stop = core.Epoch.from_array(peaks, stops)
    stop_post = core.Epoch.from_array(stops, stops + peak_stop_dur)

    get_modulation = lambda e: neurons.get_modulation_in_epochs(e, n_bins)

    modulation = []
    for s in range(3):
        epoch_slices = [_[s::3] for _ in (pre_start, start_peak, peak_stop, stop_post)]
        modulation.append(np.hstack([get_modulation(_) for _ in epoch_slices]))
    modulation = np.dstack(modulation).sum(axis=2)

    return modulation


def get_ripple_rate_trend(df):
    """To calculate ripple rate slope and correlation. This was

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    time = df["zt"].values
    ripple_rate = df["ripple_rate"].values

    linfit = stats.linregress(time, ripple_rate)
    return pd.DataFrame({"slope": linfit.slope, "rvalue": linfit.rvalue}, index=[0])
