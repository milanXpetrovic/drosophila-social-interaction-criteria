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

# normalization = json.load(open("../../../data/normalization.json"))
# pxpermm = json.load(open("../../../data/pxpermm/" + POP + ".json"))

def angledifference_nd(angle1: pd.Series, angle2: pd.Series) -> pd.Series:
    difference = angle2 - angle1
    adjustlow = difference < -180
    adjusthigh = difference > 180
    while any(adjustlow) or any(adjusthigh):
        difference[adjustlow] = difference[adjustlow] + 360
        difference[adjusthigh] = difference[adjusthigh] - 360
        adjustlow = difference < -180
        adjusthigh = difference > 180
    return difference

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

    # norm = normalization[group_name]

    for fly1, fly2 in combinations:
        df1 = group_dfs[fly1].copy(deep=True)
        df2 = group_dfs[fly2].copy(deep=True)
        df = pd.DataFrame()

        # df['movement_df1'] = ((df1['pos x'] - df1['pos x'].shift())
        #                       ** 2 + (df1['pos y'] - df1['pos y'].shift())**2)**0.5

        # df.loc[0, 'movement_df1'] = df.loc[1, 'movement_df1']

        # df['movement_df1'] = df['movement_df1']/pxpermm[group_name]/FPS
        # n, c = np.histogram(df['movement_df1'].values,
        #                     bins=np.arange(0, 2.51, 0.01))
        # opp = np.max(n) - n
        # peaks, properties = find_peaks(opp/np.max(opp), prominence=0.05)
        # movecut_df1 = 0 if len(peaks) == 0 else c[peaks[0]]

        # df2["pos x"] = (df2["pos x"] - norm["x"])  # / norm["radius"]
        # df2["pos y"] = (df2["pos y"] - norm["y"])  # / norm["radius"]
        # df1["pos x"] = (df1["pos x"] - norm["x"])  # / norm["radius"]
        # df1["pos y"] = (df1["pos y"] - norm["y"])  # / norm["radius"]

        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))

        df['distance'] = df['distance'] / (df1.a.mean()*4)
        df['distance'] = round(df['distance'], 3)

        df["dfx"] = df2['pos x'] - df1['pos x']
        df["dfy"] = df2['pos y'] - df1['pos y']

        df['checkang'] = np.arctan2(df["dfy"], df["dfx"])
        df['checkang'] =  df['checkang']*180/np.pi

        df["angle_diff"] = angledifference_nd(df["checkang"],df1["ori"]*180/np.pi)

        df["angle"] = np.round(df["angle_diff"])
        df = df[["angle", "distance"]]
        df = df[df.distance <= 10]
        #%df = df[df.movement_df1 > (movecut_df1*pxpermm[group_name]/FPS)]

        total = pd.concat([total, df], axis=0)

    total.to_csv("{}/{}.csv".format(OUTPUT_PATH, group_name))

#%%
group = fileio.load_files_from_folder(OUTPUT_PATH, file_format='.csv')

degree_bins = np.arange(-177.5, 177.6, 5)
distance_bins = np.arange(0.125, 5.8751, 0.25)
res = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

for name, path in group.items():
    df = pd.read_csv(path, index_col=0)

    hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
        degree_bins, distance_bins), range = [[-180, 180], [0, 100.0]])

    norm_hist = np.ceil((hist / np.max(hist)) * 256)
    # norm_hist = norm_hist.T
    res += norm_hist

#%%
degree_bins = np.linspace(-180, 180, 72)
distance_bins = np.linspace(0, 6, 24)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
img = ax.pcolormesh(np.radians(degree_bins), distance_bins, res.T, cmap="jet")
ax.set_rgrids(np.arange(0, 6.251, 1.0), angle=0)
ax.grid(True)
plt.title("")
plt.tight_layout()
plt.show()




#%%
group = fileio.load_files_from_folder(OUTPUT_PATH, file_format='.csv')

degree_bins = np.arange(-180, 181, 5)
distance_bins = np.arange(0, 100.051, 0.25)
res = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

for name, path in group.items():
    df = pd.read_csv(path, index_col=0)

a = df.angle.values.tolist()

f = open("angles.csv", "r")
aa = f.read()
aa = [float(x) for x in aa.split(",")]

a = [round(x) for x in a]
aa = [round(x) for x in aa]

a.sort()
aa.sort()

val, not_val = 0, 0

print(a==aa)
i = 0
for a_i, aa_i in zip(a, aa):
    i +=1
    if a_i == aa_i:
        val+=1

    else:
        not_val+=1

print(val, not_val)

df = pd.DataFrame({"a":a, "aa":aa})

#%%
import random 

start = random.randint(20, len(df)-20)
sample = df.iloc[start:start+20]

print(sample)
#%%

group = fileio.load_files_from_folder(OUTPUT_PATH, file_format='.csv')

degree_bins = np.arange(-180, 181, 5)
distance_bins = np.arange(0, 100.051, 0.25)
res = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

for name, path in group.items():
    df = pd.read_csv(path, index_col=0)
    # df.loc[df['angle'] < 0, 'angle'] += 360

    hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
        degree_bins, distance_bins), weights=df['counts'])

    norm_hist = np.ceil((hist / np.max(hist)) * 256)
    norm_hist = norm_hist.T
    # res += norm_hist

#PLOT HEATMAP
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
img = ax.pcolormesh(np.radians(degree_bins), distance_bins, res.T, cmap="jet")
ax.set_rgrids(np.arange(0, 6.251, 1.0), angle=0)
ax.grid(True)
plt.title("MOVECUT * pxpermm[group_name] / FPS")
plt.tight_layout()
plt.show()
