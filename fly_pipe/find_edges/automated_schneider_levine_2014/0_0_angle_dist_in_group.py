# %%
from typing import Dict, Tuple
import random
import os
import sys
import json
import time
import itertools
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from fly_pipe.utils import fileio


def pick_random_group(treatment: Dict[str, str], group_size: int) -> Dict[str, str]:
    """Randomly picks a group of flies from each treatment."""

    n = len(treatment)-1
    m = group_size

    picked_groups = random.sample(range(n + 1), m)
    picked_groups.sort()
    picked_flies = [random.choice(range(m)) for _ in range(m)]

    random_pick = {}
    for group_i, fly_i in zip(picked_groups, picked_flies):
        group_name = list(treatment.keys())[group_i]
        group_path = list(treatment.values())[group_i]

        group = fileio.load_files_from_folder(group_path, file_format='.csv')
        fly_name = list(group.keys())[fly_i]
        fly_name = fly_name.replace(".csv", "")
        fly_path = list(group.values())[fly_i]

        random_pick.update({"{}".format(group_name): fly_path})

    return random_pick


def angledifference_nd(angle1, angle2):
    """Calculates the difference between two angles in degrees."""

    difference = angle2 - angle1
    adjustlow = difference < -180
    adjusthigh = difference > 180
    while any(adjustlow) or any(adjusthigh):
        difference[adjustlow] = difference[adjustlow] + 360
        difference[adjusthigh] = difference[adjusthigh] - 360
        adjustlow = difference < -180
        adjusthigh = difference > 180

    return difference


def normalize_random_group(random_group: Dict[str, str],
                           normalization: Dict[str, Dict[str, float]],
                           pxpermm: Dict[str, float]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Normalize the data in each CSV file in `random_group` using the normalization values in `normalization` 
    and `pxpermm`.
    """

    normalized_dfs = {}
    pxpermm_dict = {}
    for group, fly_path in random_group.items():
        norm = normalization[group]
        pxpermm_group = pxpermm[group] / (2 * norm["radius"])

        df = pd.read_csv(fly_path, index_col=0)

        df["pos x"] = (df["pos x"] - norm["x"] +
                       norm["radius"]) / (2 * norm["radius"])
        df["pos y"] = (df["pos y"] - norm["y"] +
                       norm["radius"]) / (2 * norm["radius"])

        df["a"] = df["a"] / (2*norm["radius"])

        normalized_dfs.update({group: df})
        pxpermm_dict.update({group: pxpermm_group})

    return (normalized_dfs, pxpermm)


def group_space_angle_hist(group: Dict[str, str], norm: Dict[str, float], pxpermm_group: float) -> np.ndarray:
    """
    Generate a 2D histogram of the angular and spatial distribution of flies within the group.

    Args:
        group (Dict[str, str]): A dictionary mapping fly names to their corresponding file paths.
        norm (Dict[str, float]): A dictionary containing the x and y coordinates of the center of the arena, and the 
                                 radius of the arena, in pixels. Used for normalization of data.
        pxpermm_group (float): Pixel to millimeter conversion factor for the group.

    Returns:
        np.ndarray: A 2D histogram of the angular and spatial distribution of flies within the group.
                    The values are normalized to 256.
    """

    group_dfs = {fly: pd.read_csv(path, index_col=0)
                 for fly, path in group.items()}

    degree_bins = np.arange(-177.5, 177.6, 5)
    distance_bins = np.arange(0.125, 99.8751, 0.25)
    total = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

    for fly1, fly2 in list(itertools.permutations(group.keys(), 2)):
        df1 = group_dfs[fly1].copy(deep=True)
        df2 = group_dfs[fly2].copy(deep=True)
        df = pd.DataFrame()

        df['movement'] = ((df1['pos x'] - df1['pos x'].shift())
                          ** 2 + (df1['pos y'] - df1['pos y'].shift())**2)**0.5
        df.loc[0, 'movement'] = df.loc[1, 'movement']
        df['movement'] = df['movement']/pxpermm_group/FPS

        n, c = np.histogram(df['movement'].values,
                            bins=np.arange(0, 2.51, 0.01))

        peaks, _ = find_peaks(
            (np.max(n) - n) / np.max(np.max(n) - n), prominence=0.05)
        movecut = 0 if len(peaks) == 0 else c[peaks[0]]

        df2["pos x"] = (df2["pos x"] - norm["x"] +
                        norm["radius"]) / (2 * norm["radius"])
        df2["pos y"] = (df2["pos y"] - norm["y"] +
                        norm["radius"]) / (2 * norm["radius"])
        df1["pos x"] = (df1["pos x"] - norm["x"] +
                        norm["radius"]) / (2 * norm["radius"])
        df1["pos y"] = (df1["pos y"] - norm["y"] +
                        norm["radius"]) / (2 * norm["radius"])

        df1["a"] = df1["a"] / (2*norm["radius"])

        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))
        df['distance'] = round(df['distance'] / (df1.a.mean()*4), 4)

        df['checkang'] = np.arctan2(
            df2['pos y'] - df1['pos y'], df2['pos x'] - df1['pos x'])*180/np.pi
        df["angle"] = np.round(angledifference_nd(
            df["checkang"], df1["ori"]*180/np.pi))

        df = df[df.distance <= 100]
        df = df[df.movement > (movecut*pxpermm_group/FPS)]

        hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
            degree_bins, distance_bins), range=[[-180, 180], [0, 100.0]])

        total += hist

    norm_total = np.ceil((total / np.max(total)) * 256)

    return norm_total


def random_group_space_angle_hist(normalized_dfs: dict[str, pd.DataFrame], pxpermm: dict[str, float]) -> np.ndarray:
    """Calculate and return a 2D histogram of the angular and distance differences between pairs of flies based on their positions, using normalized dataframes.

    Args:
    normalized_dfs (Dict[str, pd.DataFrame]): A dictionary containing the normalized dataframes of the flies.
    pxpermm (Dict[str, float]): A dictionary containing the conversion factor from pixels to millimeters for each fly.

    Returns:
    A numpy array representing the normalized 2D histogram of the angular and distance differences between pairs of flies.
    """

    degree_bins = np.arange(-177.5, 177.6, 5)
    distance_bins = np.arange(0.125, 99.8751, 0.25)
    total = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

    for fly1_key, fly2_key in list(itertools.permutations(normalized_dfs.keys(), 2)):
        df1 = normalized_dfs[fly1_key].copy(deep=True)
        df2 = normalized_dfs[fly2_key].copy(deep=True)
        df = pd.DataFrame()

        df['movement'] = ((df1['pos x'] - df1['pos x'].shift())
                          ** 2 + (df1['pos y'] - df1['pos y'].shift())**2)**0.5
        df.loc[0, 'movement'] = df.loc[1, 'movement']

        df['movement'] = df['movement']/pxpermm[fly1_key]/FPS

        n, c = np.histogram(df['movement'].values,
                            bins=np.arange(0, 2.51, 0.01))

        peaks, _ = find_peaks(
            (np.max(n) - n) / np.max(np.max(n) - n), prominence=0.05)
        movecut = 0 if len(peaks) == 0 else c[peaks[0]]

        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))
        df['distance'] = round(df['distance'] / (df1.a.mean()*4), 4)

        df['checkang'] = np.arctan2(
            df2['pos y'] - df1['pos y'], df2['pos x'] - df1['pos x'])*180/np.pi
        df["angle"] = np.round(angledifference_nd(
            df["checkang"], df1["ori"]*180/np.pi))

        df = df[df.distance <= 100]
        df = df[df.movement > (movecut*pxpermm[fly1_key]/FPS)]

        hist, _, _ = np.histogram2d(df['angle'], df['distance'], bins=(
            degree_bins, distance_bins), range=[[-180, 180], [0, 100.0]])

        total += hist

    norm_total = np.ceil((total / np.max(total)) * 256)

    return norm_total


def plot_heatmap(histogram):
    """
    NEed to fix this foo. 
    add code that calculates MAX_DIST  
    """

    degree_bins = np.linspace(-180, 180, 71)
    distance_bins = np.linspace(0, 6, 24)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    img = ax.pcolormesh(np.radians(degree_bins),
                        distance_bins, histogram, cmap="jet")
    ax.set_rgrids(np.arange(0, 6.251, 1.0), angle=0)
    ax.grid(True)
    plt.title("")
    plt.tight_layout()
    plt.show()


FPS = 22.8
POP = "CSf"
INPUT_PATH = "../../../data/raw/" + POP
OUTPUT_PATH = "../../../data/find_edges/0_0_angle_dist_in_group/" + POP

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

normalization = json.load(open("../../../data/normalization.json"))
pxpermm = json.load(open("../../../data/pxpermm/" + POP + ".json"))

treatment = fileio.load_multiple_folders(INPUT_PATH)

# %%
for group_name, group_path in treatment.items():
    print(group_name)
    norm = normalization[group_name]
    pxpermm_group = pxpermm[group_name] / (2 * norm["radius"])
    group = fileio.load_files_from_folder(group_path, file_format='.csv')
    hist = group_space_angle_hist(group, norm, pxpermm_group)

    np.save("{}/{}".format(OUTPUT_PATH, group_name), hist)

# %%
group = fileio.load_files_from_folder(OUTPUT_PATH, file_format='.npy')
res = np.sum([np.load(path) for path in group.values()], axis=0)

# %%


def test_np_hist(normalized_dfs: dict[str, pd.DataFrame], pxpermm: dict[str, float]) -> np.ndarray:

    degree_bins = np.arange(-177.5, 177.6, 5)
    distance_bins = np.arange(0.125, 99.8751, 0.25)
    total = np.zeros((len(degree_bins)-1, len(distance_bins)-1))

    for fly1_key, fly2_key in list(itertools.permutations(normalized_dfs.keys(), 2)):
        df1 = normalized_dfs[fly1_key].copy(deep=True)
        df2 = normalized_dfs[fly2_key].copy(deep=True)

        df1_array = df1.to_numpy()
        df2_array = df2.to_numpy()

        movement = np.sqrt((df1_array[:, 0] - np.roll(df1_array[:, 0], 1))**2
                           + (df1_array[:, 1] - np.roll(df1_array[:, 1], 1))**2)
        movement[0] = movement[1]
        movement = movement / pxpermm[fly1_key] / FPS

        n, c = np.histogram(movement, bins=np.arange(0, 2.51, 0.01))

        peaks, _ = find_peaks(
            (np.max(n) - n) / np.max(np.max(n) - n), prominence=0.05)
        movecut = 0 if len(peaks) == 0 else c[peaks[0]]

        a = np.mean(df1_array[:, 3])
        distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0])**2
                           + (df1_array[:, 1] - df2_array[:, 1])**2)
        distance = np.round(distance / (a * 4), 4)

        checkang = np.arctan2(
            df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])
        checkang = checkang * 180 / np.pi

        angle = angledifference_nd(checkang, df1_array[:, 2]*180/np.pi)
        angle = np.round(angle)

        df = pd.DataFrame(
            {'angle': angle, 'distance': distance, 'movement': movement})

        mask = (distance <= 100) & (movement > (
            movecut * pxpermm[fly1_key] / FPS))
        angle = angle[mask]
        distance = distance[mask]

        hist, _, _ = np.histogram2d(angle, distance, bins=(
            degree_bins, distance_bins), range=[[-180, 180], [0, 100.0]])

        total += hist

    norm_total = np.ceil((total / np.max(total)) * 256)

    return norm_total


all_hists = []
for i in range(1):
    t1 = time.time()
    random_group = pick_random_group(treatment, group_size=12)
    t2 = time.time()
    print(f"Time taken for pick_random_group: {t2-t1:.2f} seconds")

    t1 = time.time()
    normalized_dfs, pxpermm = normalize_random_group(
        random_group, normalization, pxpermm)
    t2 = time.time()
    print(f"Time taken for normalize_random_group: {t2-t1:.2f} seconds")

    t1 = time.time()
    hist_df = random_group_space_angle_hist(normalized_dfs, pxpermm)
    t2 = time.time()
    print(f"Time taken for random_group_space_angle_hist: {t2-t1:.2f} seconds")

    t1 = time.time()
    hist_np = test_np_hist(normalized_dfs, pxpermm)
    t2 = time.time()
    print(f"Time taken for test_np_hist: {t2-t1:.2f} seconds")

    all_hists.append(hist)

# %%


# %%
res = np.sum(all_hists, axis=0)
res = res.T
res = res[:24]
plot_heatmap(res)
