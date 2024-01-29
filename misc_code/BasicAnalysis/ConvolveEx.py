import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
y = signal.square(2 * np.pi * 1 * (t-0.25))

t2 = np.linspace(0.05, 0.2, 200)
y2 = signal.square(2 * np.pi * 7 * (t2-0.8))

c1 = np.convolve(y2, y, mode='same')

d = 4

plt.clf()

plt.subplot(3, 1, 1)
plt.plot(t, y)

plt.subplot(3, 1, 2)
plt.plot(t2, y2)

plt.subplot(3, 1, 3)
plt.plot(t, c1)

plt.show()
