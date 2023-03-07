# %%
from mpl_toolkits.mplot3d import Axes3D
import sys
import itertools
import pandas as pd
import numpy as np
from fly_pipe.utils import fileio
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np

PATH = "../../../data/raw\CSf"
treatment = fileio.load_multiple_folders(PATH)

for group_name, group_path in treatment.items():
    print(group_name)
    group = fileio.load_files_from_folder(group_path, file_format='.csv')

    total = pd.DataFrame()
    # group_dfs = {fly: pd.read_csv(path, index_col=0)
    #              for fly, path in group.items()}

    combinations = list(itertools.permutations(group.keys(), 2))
    print(len(combinations))
    for fly1, fly2 in combinations:
        df1 = pd.read_csv(group[fly1], index_col=0)
        df2 = pd.read_csv(group[fly2], index_col=0)

        df = pd.DataFrame()
        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))

        df['distance'] = df['distance'] / (df1.a.mean()*4)
        df['distance'] = round(df['distance'], 2)

        df2["pos x"] = df2["pos x"] - df1["pos x"]
        df2["pos y"] = df2["pos y"] - df1["pos y"]

        df1["pos x"] = df1["pos x"] - df1["pos x"]
        df1["pos y"] = df1["pos y"] - df1["pos y"]

        df["dfx"] = df2['pos x'] - df1['pos x']
        df["dfy"] = df2['pos y'] - df1['pos y']

        df1.loc[df1['ori'] < 0, 'ori'] *= -1  # 2*np.pi

        cos = np.cos(df1["ori"])
        sin = np.sin(df1["ori"])

        df['qx2'] = df1['pos x'] + cos * df["dfx"] - sin * df["dfy"]
        df['qy2'] = df1['pos y'] + sin * df["dfx"] + cos * df["dfy"]

        # df['angle'] = np.arctan2(
        #     df["qy2"]-df1["pos y"], df["qx2"]-df1["pos x"])
        df['angle'] = np.arctan2(df["qy2"], df["qx2"])
        df['angle'] = np.rad2deg(df['angle'])
        df['angle'] = np.round(df['angle'])

        df = df[df.distance <= 20]
        df = df.groupby(['angle', 'distance']
                        ).size().reset_index(name='counts')
        # df = df[['angle', 'distance']]

        total = pd.concat([total, df], axis=0)
    total.to_csv(
        "../../../data/find_edges/0_0_angle_dist_in_group/CSf/" + group_name + ".csv")

group = fileio.load_files_from_folder(
    "../../../data/find_edges/0_0_angle_dist_in_group/CSf/", file_format='.csv')
tot = pd.DataFrame()
for name, path in group.items():
    df = pd.read_csv(path, index_col=0)
    tot = pd.concat([tot, df], axis=0)

df = tot.groupby(['angle', 'distance'])[
    'counts'].sum().reset_index(name='counts')

# define the bins
degree_bins = np.arange(-180, 181, 5)
distance_bins = np.arange(0, 20.001, 0.25)

# calculate the 2D histogram
hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
    degree_bins, distance_bins), weights=df['counts'])

# plot the histogram as a polar heatmap
# plt.style.use('dark_background')  # this makes the text and grid lines white
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'polar': True})
ax.grid(False)  # pcolormesh gives an annoying warning when the grid is on
ax.pcolormesh(np.radians(degree_bins), distance_bins, hist.T, cmap='jet')
ax.grid(True)
plt.tight_layout()
plt.show()

# %%
degree_bins = np.arange(-180, 181, 5)
distance_bins = np.arange(0, 6.001, 0.25)
round_bins = pd.DataFrame(0, index=distance_bins, columns=degree_bins)

for row in df.iterrows():
    _, value = row
    degree = value["angle"]
    distance = value["distance"]
    count = value["counts"]
    degree_bin = np.digitize([degree], degree_bins)[0]-1
    distance_bin = np.digitize([distance], distance_bins)[0]-1
    round_bins.iloc[distance_bin, degree_bin] += count
n = len(degree_bins)
m = len(distance_bins)

rad = np.linspace(0, 10., m)
a = np.linspace(0, 2 * np.pi, n)
r, th = np.meshgrid(rad, a)
z = round_bins.to_numpy().T
plt.figure(figsize=(10, 10))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap='jet')
plt.plot(a, r, ls='none', color='k')
plt.grid()
plt.colorbar()


# %%
histogram = round(round_bins.sum(axis=1))
histogram.plot(kind="bar")

# TODO:
# fix pxpermm and major x axis (column "a") for each individual
# make ploting to work as function
# make distribution to rounds bins as function
