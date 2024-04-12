import numpy as np
from neuropy.utils.signal_process import hilbertfast, filter_sig
from neuropy.core import Signal, Position
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def theta_speed_ccg(signal: Signal, position: Position):

    fs = signal.sampling_rate
    pos_srate = position.sampling_rate
    pos_time = position.time
    speed = position.speed
    lfp_time = signal.time
    signal_dur = lfp_time[-1] - lfp_time[0]
    smooth = lambda arr: gaussian_filter1d(arr, 0.01 / (1 / pos_srate))

    xcorrs = []
    for trace in signal.traces:

        hilamp = np.abs(
            hilbertfast(filter_sig.bandpass(trace, lf=5, hf=12, fs=fs).traces[0])
        )
        # delta= np.abs(hilbertfast(filter_sig.bandpass(lfp,lf=0.4,hf=16).traces[0]))

        amp_zsc = stats.zscore(hilamp)
        speed_zsc = stats.zscore(speed)
        hilamp_ds = np.interp(pos_time, lfp_time, amp_zsc)
        # hilamp_ds = smooth(hilamp_ds)
        # speed_zsc = smooth(speed_zsc)
        # speed_zsc = np.roll(speed_zsc,0)

        xcorr = np.correlate(speed_zsc, hilamp_ds, mode="same")
        xcorr = stats.zscore(xcorr)
        t = np.linspace(-signal_dur / 2, signal_dur / 2, len(xcorr))
        indices = (t > -4) & (t < 4)
        xcorrs.append(xcorr[indices])

    _,ax =plt.subplots()
    cmap = get_cmap('jet')
    for i,x in enumerate(xcorrs):
        dur = len(x)/pos_srate
        t = np.linspace(-dur/2,dur/2,len(x))
        ax.plot(t,x,color=cmap(i/len(xcorrs)))
        ax.axvline(t[np.argmax(x)],color='r',ls='--')
        print(t[np.argmax(x)])

    ax.axvline(0,color='k',ls='--')
    ax.set_xlim(-4,4)
