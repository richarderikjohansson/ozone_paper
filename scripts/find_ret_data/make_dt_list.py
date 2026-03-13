from ozone.io import get_data_files_root
import numpy as np
from datetime import datetime, timedelta


all_files = np.loadtxt("files.txt", dtype=str)
ret_files = np.loadtxt("ret_files.txt", dtype=str)
new_file_list = list()

for file in all_files:
    fn = file.split("/")[-1]
    if fn in ret_files:
        new_file_list.append(file)

with open("new_file_list.txt", "w") as fh:
    for file in new_file_list:
        fh.write(f"{file}\n")
