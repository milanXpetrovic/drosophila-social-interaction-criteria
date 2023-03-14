# TODO:
# fix pxpermm and major x axis (column "a") for each individual
# make ploting to work as function
# make distribution to rounds bins as function

import os
from fly_pipe.utils import schneider_social as ss
from fly_pipe.utils import fileio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import time
import json

import pandas as pd


def find_distances_and_angles_in_group(group, pxpermm, OUTPUT_PATH):
    """
    group - dictionary of paths and filenames to .csv files
    group : dict {file_name : file_path}
    """

    total = pd.DataFrame()
    group_dfs = {fly: pd.read_csv(path) for fly, path in group.items()}
    combinations = list(itertools.permutations(group_dfs.keys(), 2))

    for fly1, fly2 in combinations:
        df1, df2 = group_dfs[fly1], group_dfs[fly2]

        df = pd.DataFrame()
        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))
        df['distance'] = df['distance'] / df1.a.mean()
        df['distance'] = df['distance'] / (pxpermm)
        df['distance'] = round(df['distance'], 2)

        df = df[df.distance <= 20]

        df['qx2'] = df1['pos x'] + np.cos(df1['ori']) * (df2['pos x'] -
                                                         df1['pos x']) - np.sin(df1['ori']) * (df2['pos y'] - df1['pos y'])
        df['qy2'] = df1['pos y'] + np.sin(df1['ori']) * (df2['pos x'] -
                                                         df1['pos x']) + np.cos(df1['ori']) * (df2['pos y'] - df1['pos y'])

        df['angle'] = round(np.rad2deg(np.arctan2(
            (df['qy2']-df1['pos y']), (df['qx2']-df1['pos x']))))

        df = df.groupby(
            ['angle', 'distance']).size().reset_index(name='counts')

        total = pd.concat([total, df], axis=0)

    return total


path = "../../data/find_edges/0_0_angle_dist/csf/"
group = fileio.load_files_from_folder(path, file_format='.csv')
tot = pd.DataFrame()
for name, path in group.items():
    df = pd.read_csv(path, index_col=0)
    df = df.groupby(['angle', 'distance'])[
        'counts'].sum().reset_index(name='counts')
    tot = pd.concat([tot, df], axis=0)

summ = tot.groupby(['angle', 'distance'])[
    'counts'].sum().reset_index(name='counts')

# %%

min_val = summ['counts'].min()
max_val = summ['counts'].max()
summ['counts'] = (summ['counts'] - min_val) / (max_val - min_val)

degree_bins = np.array([x for x in range(-180, 181, 5)])
distance_bins = np.array([x*0.25 for x in range(0, 81)])
round_bins = pd.DataFrame(0, index=distance_bins, columns=degree_bins)

for row in summ.iterrows():
    _, value = row
    degree = value["angle"]
    distance = value["distance"]
    count = value["counts"]
    degree_bin = np.digitize([degree], degree_bins)[0]-1
    distance_bin = np.digitize([distance], distance_bins)[0]-1
    round_bins.iloc[distance_bin, degree_bin] += count

n = len(degree_bins)
m = len(distance_bins)

rad = np.linspace(0, 20.25, m)
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


def plot_spatial_heatmap(bins, DISTANCE_THRESHOLD, ROUND_ANGLE=5, ROUND_DIST=1, ):
    n = int((360/ROUND_ANGLE) + 1)
    m = int((DISTANCE_THRESHOLD / ROUND_DIST) + 1)

    rad = np.linspace(0, 10, m)
    a = np.linspace(0, 2 * np.pi, n)
    r, th = np.meshgrid(rad, a)
    z = bins.to_numpy().T

    plt.figure(figsize=(10, 10))
    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, z, cmap='jet')
    plt.plot(a, r, ls='none', color='k')
    plt.grid()
    plt.colorbar()
    plt.show()
