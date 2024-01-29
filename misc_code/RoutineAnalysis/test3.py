import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.fftpack as fft
import scipy.fft as sf
import scipy.signal as sg
import scipy.ndimage as filtSig
from scipy.interpolate import interp1d
from signal_process import spectrogramBands
from callfunc import processData


def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    # print(freqs[:20])
    # freqs1 = np.linspace(0, 2048.0, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


basePath = "/data/Clustering/SleepDeprivation/RatN/Day1/"


sess = processData(basePath)

lfp = np.load(sess.sessinfo.files.thetalfp)[: 1000 * 1250]
N = len(lfp)
# sample spacing
T = 1.0 / 1250.0
t = np.linspace(0, N * T, N)
yf = fft.fft(lfp)
xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
y = 2.0 / N * np.abs(yf[: N // 2])
# y = filtSig.gaussian_filter1d(y, 2, axis=0)

fs = 1250
freqs, Pxx = sg.welch(lfp, fs=fs, nperseg=2 * 1250)
psd = interp1d(freqs, Pxx)
sig_whiten = whiten(lfp, psd, T)
yf2 = np.fft.rfft(sig_whiten)
y2 = 2.0 / N * np.abs(yf2)

plt.clf()
fig = plt.figure(1, figsize=(6, 10))
gs = GridSpec(5, 1, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, lfp)

ax1 = fig.add_subplot(gs[1, 0])
ylog = np.log10(y)
ax1.plot(xf, y)


ax2 = fig.add_subplot(gs[2, 0])
# ydet = sg.detrend(ylog, type="linear")
# ynew = 10 ** ydet
ax2.plot(y2)

ax3 = fig.add_subplot(gs[3, 0])
# signew = fft.ifft(np.concatenate((ynew, np.flip(ynew))))

ax3.plot(sig_whiten)

ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
m = spectrogramBands(lfp, window=625)
sxx = m.sxx * 1000
ax4.pcolormesh(m.time, m.freq, sxx[:, :], cmap="YlGn", vmax=800500, vmin=1000)
