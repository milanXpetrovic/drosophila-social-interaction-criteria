# %%
import sys
import json
import itertools
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fly_pipe.utils import fileio


FPS = 22.8
POP = "CSf"
PATH = "../../../data/raw/" + POP
OUTPUT_PATH = "../../../data/find_edges/0_0_angle_dist_in_group/" + POP

normalization = json.load(open("../../../data/normalization.json"))
pxpermm = json.load(open("../../../data/pxpermm/" + POP + ".json"))

treatment = fileio.load_multiple_folders(PATH)
for group_name, group_path in treatment.items():
    print(group_name)
    group = fileio.load_files_from_folder(group_path, file_format='.csv')

    total = pd.DataFrame()
    group_dfs = {fly: pd.read_csv(path, index_col=0)
                 for fly, path in group.items()}
    combinations = list(itertools.permutations(group.keys(), 2))

    norm = normalization[group_name]

    # for fly in group.keys():
    #     df1 = group_dfs[fly]

    for fly1, fly2 in combinations:
        df1 = group_dfs[fly1].copy(deep=True)
        df2 = group_dfs[fly2].copy(deep=True)
        df = pd.DataFrame()

        # df['movement'] = ((df1['pos x'] - df1['pos x'].shift())
        #                   ** 2 + (df1['pos y'] - df1['pos y'].shift())**2)**0.5

        # df['movement'] = df['movement']/pxpermm[group_name]/FPS
        # df.loc[0, 'movement'] = df.loc[1, 'movement']

        # n, c = np.histogram(df['movement'].values,
        #                     bins=np.arange(0, 2.51, 0.01))
        # opp = np.max(n) - n
        # peaks, properties = find_peaks(opp/np.max(opp), prominence=0.05)

        # if len(peaks) == 0:
        #     movecut = 0

        # else:
        #     movecut = c[peaks[0]]

        df2["pos x"] = (df2["pos x"] - norm["x"])  # / norm["radius"]
        df2["pos y"] = (df2["pos y"] - norm["y"])  # / norm["radius"]
        df1["pos x"] = (df1["pos x"] - norm["x"])  # / norm["radius"]
        df1["pos y"] = (df1["pos y"] - norm["y"])  # / norm["radius"]

        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))

        df['distance'] = df['distance'] / (df1.a.mean()*4)
        df['distance'] = round(df['distance'], 2)

        df1.loc[df1['ori'] < 0, 'ori'] *= -1

        df2["pos x"] = df2["pos x"] - df1["pos x"]
        df2["pos y"] = df2["pos y"] - df1["pos y"]

        df1["pos x"] = df1["pos x"] - df1["pos x"]
        df1["pos y"] = df1["pos y"] - df1["pos y"]

        df["dfx"] = df2['pos x'] - df1['pos x']
        df["dfy"] = df2['pos y'] - df1['pos y']

        cos = np.cos(df1["ori"])
        sin = np.sin(df1["ori"])

        df['qx2'] = df1['pos x'] + cos * df["dfx"] - sin * df["dfy"]
        df['qy2'] = df1['pos y'] + sin * df["dfx"] + cos * df["dfy"]

        df['angle'] = np.arctan2(df["qy2"], df["qx2"])
        df['angle'] = np.rad2deg(df['angle'])
        df['angle'] = np.round(df['angle'])

        df = df[df.distance <= 7]
        # df = df[df.movement >= movecut]

        df = df.groupby(['angle', 'distance']
                        ).size().reset_index(name='counts')

        total = pd.concat([total, df], axis=0)

    total.to_csv("{}/{}.csv".format(OUTPUT_PATH, group_name))

group = fileio.load_files_from_folder(OUTPUT_PATH, file_format='.csv')

degree_bins = np.arange(0, 361, 5)
distance_bins = np.arange(0, 6.251, 0.25)
res = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

for name, path in group.items():
    df = pd.read_csv(path, index_col=0)
    df.loc[df['angle'] < 0, 'angle'] += 360

    hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
        degree_bins, distance_bins), weights=df['counts'])

    norm_hist = np.ceil((hist / np.max(hist)) * 256)
    res += hist

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
img = ax.pcolormesh(np.radians(degree_bins), distance_bins, res.T, cmap='jet')

ax.set_rgrids(np.arange(0, 6.251, 0.5), angle=0)
ax.grid(True)

plt.tight_layout()
plt.show()
