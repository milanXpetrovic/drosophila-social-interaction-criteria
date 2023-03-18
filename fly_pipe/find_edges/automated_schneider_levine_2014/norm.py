# %%
import sys
import matplotlib.pyplot as plt
import json
import scipy.io
from fly_pipe.utils import fileio

INPUT_PATH = r"C:/Users/icecream/Desktop/sch_data/raw/g11"
group = fileio.load_files_from_folder(INPUT_PATH, file_format='.mat')

d = {}
for name, path in group.items():
    m = scipy.io.loadmat(path)

    m1 = m["trx"]
    m2 = m["trx"][0][0][6]
    pxpermm = m2[0][0]

    d.update({name.replace(".mat", ""): pxpermm})


with open("pox-neural.json", "w") as outfile:
    json.dump(d, outfile, indent=4)


# %%
m = mat_data['db_n'][0]
d = {}
for i in range(len(m)):
    circle, name = mat_data['db_n'][0][i]
    circle = circle[0]
    x, y, radius = circle[0], circle[1], circle[2]

    d.update({name[0].replace(".mat", ""): {"x": x, "y": y, "radius": radius}})


# %%
PATH = "F:/fly-pipe/data/normalization.json"

d = json.load(open(PATH))

with open("normalization.json", "w") as outfile:
    json.dump(d, outfile, indent=4)
