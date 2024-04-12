import numpy as np
import matplotlib.pyplot as plt
import pywt
import math
from scipy import signal


t = np.linspace(0, 0.6, 4000)
# s = np.sin(8 * 2 * np.pi * t) + 0.5*np.sin(16 * 2 *
#                                            np.pi * t) + 0.3*np.sin(30 * 2 * np.pi * t)

s = np.sin(8 * 2 *
           np.pi * t) + 0.5*np.sin(16 * 2 * np.pi * t) + 0.1*np.random.randn(len(t))
dt = t[1]-t[0]


# # fft analysis
# fft = np.fft.fft(s)
# T = t[1] - t[0]  # sampling interval
# N = s.size
# # 1/T = frequency
# f = np.linspace(0, 1 / T, N)

# == wavelet analysis ==========================
time, sst = t, s


C_range = np.arange(1, 71, 1)
# scales1 = np.arange(1, 100, 1)
# f2 = pywt.scale2frequency(wavelet, scales1)
spect = np.zeros(shape=(len(C_range), 4000))
for wav in range(0, len(C_range)):

    C = C_range[wav]
    B = 2*((7/(2*np.pi*C))**2)
    wavelet = 'cmor' + repr(B) + '-' + repr(C)
    sigma_t = 7/(2*np.pi*C)
    # scales = np.arange(0.5, 60, 0.01)

    A = (sigma_t * np.sqrt(np.pi)) ** (-1/2)
    gauss_env = np.exp((-(t-0.3)**2)/(2*sigma_t**2))
    # f2 = pywt.scale2frequency(wavelet, scales)
    mor_wavelet_real = A * np.multiply(gauss_env, np.cos(2*np.pi*C*t))
    mor_wavelet_img = A * np.multiply(gauss_env, np.sin(2*np.pi*C*t))
    cfs1 = np.convolve(s, mor_wavelet_real, mode='same')**2
    cfs2 = np.convolve(s, mor_wavelet_img, mode='same')**2
    cfs = cfs1+cfs2
    # [cfs, frequencies] = pywt.cwt(sst, 600, wavelet)
    # logpower = np.log10((abs(cfs)) ** 2)
    logpower = np.log10(cfs)
    spect[wav] = logpower


spect = np.flip(spect)

# period = frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]

plt.clf()
plt.subplot(311)
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.plot(t, s)

# plt.subplot(412)
# plt.ylabel("Amplitude")
# plt.xlabel("Frequency [Hz]")
# # 1 / N is a normalization factor
# plt.plot(f[:N // 10], np.abs(fft)[:N // 10] * 1 / N)

plt.subplot(312)

plt.imshow(spect, aspect='auto', vmin=4,
           extent=[0, 1, 1, 70], cmap='YlOrRd')
# plt.contour(t, C_range, spect)


plt.subplot(313)
plt.plot(t, mor_wavelet_real)

# f, ax = plt.subplots(figsize=(15, 10))
# ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
#             extend='both')

# ax.set_title('%s Wavelet Power Spectrum (%s)' % ('Nino1+2', wavelet))
# ax.set_ylabel('Period (years)')
# Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
#                         np.ceil(np.log2(period.max())))
# ax.set_yticks(np.log2(Yticks))
# ax.set_yticklabels(Yticks)
# ax.invert_yaxis()
# ylim = ax.get_ylim()
# ax.set_ylim(ylim[0], -1)
