import numpy as np
from pathlib import Path
from ozone.io import get_downloadsdir

# file = get_downloadsdir() / "new_ret.npy"

m2_night_file = "mira2_matching_night.npy"
m2_day_file = "mira2_matching.npy"
mls_night_file = "mls_matching_night.npy"
m2_night = np.load(file=m2_night_file, allow_pickle=True).item()
m2_day = np.load(file=m2_day_file, allow_pickle=True).item()
mls_night = np.load(file=mls_night_file, allow_pickle=True).item()


day = list()
for val in m2_day.values():
    file = val["file"]
    day.append(file)


night = list()
for val in m2_night.values():
    file = val["file"]
    night.append(file)

files = np.concatenate((day, night))

for file in files:
    print(file)

with open("ret_files.txt", "w") as fh:
    for file in files:
        filename = file[0].name
        fh.write(f"{filename}\n")
