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

degree_bins = np.arange(0, 361, 5)
distance_bins = np.arange(0, 6.251, 0.25)

for group_name, group_path in treatment.items():
    print(group_name)
    group = fileio.load_files_from_folder(group_path, file_format='.csv')

    total = pd.DataFrame()
    group_dfs = {fly: pd.read_csv(path, index_col=0)
                 for fly, path in group.items()}
    combinations = list(itertools.permutations(group.keys(), 2))

    norm = normalization[group_name]

    for fly1, fly2 in combinations:
        df1 = group_dfs[fly1].copy(deep=True)
        df2 = group_dfs[fly2].copy(deep=True)
        df = pd.DataFrame()

        df['movement_df1'] = ((df1['pos x'] - df1['pos x'].shift())
                              ** 2 + (df1['pos y'] - df1['pos y'].shift())**2)**0.5

        df.loc[0, 'movement_df1'] = df.loc[1, 'movement_df1']

        df['movement_df1'] = df['movement_df1']/pxpermm[group_name]/FPS
        n, c = np.histogram(df['movement_df1'].values,
                            bins=np.arange(0, 2.51, 0.01))
        opp = np.max(n) - n
        peaks, properties = find_peaks(opp/np.max(opp), prominence=0.05)
        movecut_df1 = 0 if len(peaks) == 0 else c[peaks[0]]

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

        df['angle'] = np.arctan2(df["dfy"], df["dfx"])
        df['angle'] = np.rad2deg(df['angle'])
        df['angle'] = np.round(df['angle'])

        df["angle_diff"] = (np.rad2deg(df1["ori"])) - df["angle"]

        while df["angle_diff"].min() < -180 and df["angle_diff"].max() > 180:
            df.loc[df['angle_diff'] < -180, 'angle_diff'] += 360
            df.loc[df['angle_diff'] > 180, 'angle_diff'] -= 360

        df["angle"] = np.round(df["angle_diff"])

        df = df[df.distance <= 7]
        # df = df[df.movement_df1 > (movecut_df1*pxpermm[group_name]/FPS)]

        df = df.groupby(['angle', 'distance']
                        ).size().reset_index(name='counts')

        total = pd.concat([total, df], axis=0)

    # histogram and bins
    # hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
    #     degree_bins, distance_bins), weights=df['counts'])

    total.to_csv("{}/{}.csv".format(OUTPUT_PATH, group_name))

# %%
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
    res += norm_hist

res = np.ceil((res / np.max(res)) * 256)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

img = ax.pcolormesh(np.radians(degree_bins), distance_bins, res.T, cmap="jet")

ax.set_rgrids(np.arange(0, 6.251, 1.0), angle=0)
ax.grid(True)
plt.title("MOVECUT * pxpermm[group_name] / FPS")
plt.tight_layout()
plt.show()
