import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.signal as sg


def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, "same"))


t = np.arange(0, 5, 0.001)
sig = (
    np.sin(2 * np.pi * 140 * t)
    * np.sin(2 * np.pi * 4 * t)
    # * np.sin(2 * np.pi * 220 * t)
    * np.sin(2 * np.pi * 2 * t)
    * np.sin(2 * np.pi * 8 * t)
)

analytic_signal = sg.hilbert(sig)
amplitude_envelope = np.abs(analytic_signal)


rms_sig = window_rms(sig, 7)
plt.clf()
plt.plot(t, sig)
plt.plot(t, rms_sig)
plt.plot(t, amplitude_envelope)
