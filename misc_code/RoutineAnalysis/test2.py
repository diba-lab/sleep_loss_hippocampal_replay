
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
Z = np.random.rand(6, 10)

fig, (ax0, ax1) = plt.subplots(2, 1)

c = ax0.pcolor(Z)
ax0.set_title('default: no edges')

# c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
# ax1.set_title('thick edges')

# fig.tight_layout()
# plt.show()

dx, dy = 0.15, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(-3, 3 + dy, dy), slice(-3, 3 + dx, dx)]
z = (1 - x / 2.0 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, axs = plt.subplots(2, 2)

ax = axs[0, 0]
c = ax.pcolor(x, y, z, cmap="RdBu", vmin=z_min, vmax=z_max)
ax.set_title("pcolor")
fig.colorbar(c, ax=ax)

ax = axs[0, 1]
c = ax.pcolormesh(x, y, z, cmap="RdBu", vmin=z_min, vmax=z_max)
ax.set_title("pcolormesh")
fig.colorbar(c, ax=ax)

ax = axs[1, 0]
c = ax.imshow(
    z,
    cmap="RdBu",
    vmin=z_min,
    vmax=z_max,
    extent=[x.min(), x.max(), y.min(), y.max()],
    interpolation="nearest",
    origin="lower",
)
ax.set_title("image (nearest)")
fig.colorbar(c, ax=ax)

ax = axs[1, 1]
c = ax.pcolorfast(x, y, z, cmap="RdBu", vmin=z_min, vmax=z_max)
ax.set_title("pcolorfast")
fig.colorbar(c, ax=ax)

fig.tight_layout()
plt.show()
