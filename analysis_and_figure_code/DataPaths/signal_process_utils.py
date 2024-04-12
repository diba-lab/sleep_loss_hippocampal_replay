import numpy as np
from neuropy.utils.signal_process import hilbertfast, filter_sig, FourierSg
from neuropy.core import Signal
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from dip import diptst
from numpy.polynomial.polynomial import polyfit


def get_dip_stat(x, nbins=200):
    bins = np.linspace(x.min(), x.max(), nbins)
    hist_x = np.histogram(x, bins)[0]
    return diptst(hist_x, is_hist=True)[:2]


def get_sw_theta_bimodal_params(lfp):
    """Gives bimodality statistic to for a given time series in delta and theta range. The bimodality is tested on the slope in 4-90 Hz (delta) and 2-20 Hz (theta).

    Parameters
    ----------
    lfp : array
        timeseries

    Returns
    -------
    stat_delta, pval_delta, state_theta, pval_theta
    """
    signal = Signal(traces=lfp, sampling_rate=1250)
    freqs = np.geomspace(1, 100, 100)
    freqs = freqs[(freqs < 57) | (freqs > 63)]
    spect = FourierSg(signal, freqs=freqs, window=2, overlap=1)
    spect_for_sw = spect.freq_slice(4, 90)
    spect_for_theta = spect.freq_slice(2, 20)

    spect_slope = lambda sxx: polyfit(sxx.freqs, np.log10(sxx.traces), deg=1)[1]
    slope_sw = spect_slope(spect_for_sw)
    slope_theta = spect_slope(spect_for_theta)

    return *get_dip_stat(slope_sw), *get_dip_stat(slope_theta)
