import numpy as np
import matplotlib.pyplot as plt

z = np.loadtxt("z.txt")
ph = np.loadtxt("ph.txt")
pf = np.loadtxt("pf.txt")

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.invert_yaxis()
ax.plot(z, ph, label="ph")
ax.plot(z, pf, label="pf")
ax.legend()
plt.show()
