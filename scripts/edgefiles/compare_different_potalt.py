from ozone.io import get_egdefiles
from ozone.utils import parse_edgefile, filter_edgedata
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

files = get_egdefiles("/home/ric/Data/edge/")
e550 = filter_edgedata(parse_edgefile(files[0]), 1)
e475 = filter_edgedata(parse_edgefile(files[1]), 1)

combined = np.unique(np.concatenate([e550.date, e475.date]))
print(len(combined))
print(len(e550.date))
print(len(e475.date))


fig = plt.figure(figsize=(14, 4))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(combined, np.full_like(combined, 5), label="combined")
ax.scatter(e550.date, np.full_like(e550.date, 3), label="e550")
ax.scatter(e475.date, np.full_like(e475.date, 1), label="e475")
ax.legend()
plt.show()
