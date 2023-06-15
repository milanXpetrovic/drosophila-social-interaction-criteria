# %%
import itertools
import json
import multiprocessing
import os
import random
import sys
import time
import warnings
from collections import defaultdict

import natsort
import numpy as np
import pandas as pd
import scipy
from scipy.io import savemat
from scipy.ndimage import label
from scipy.signal import convolve2d, find_peaks
from skimage import measure as skimage_label

import src.utils.automated_schneider_levine as SL
from src import settings
from src.utils import fileio


def one_run_random(tuple_args):
    treatment, normalization, pxpermm = tuple_args

    random_group = SL.pick_random_group(treatment)
    normalized_dfs, pxpermm = SL.normalize_group(random_group, normalization, pxpermm)
    hist_np = SL.group_space_angle_hist(normalized_dfs, pxpermm)
    return hist_np


def fast_flag_interactions(trx, timecut, minang, bl, start, exptime, nflies, fps, movecut):
    sorted_keys = natsort.natsorted(trx.keys())

    trx = {k: trx[k] for k in sorted_keys}
    start = round(start * 60 * fps + 1)
    timecut = timecut * fps
    m = [1, 41040]
    nflies = len(trx)

    mindist = np.zeros((nflies, 1))
    i = 0
    for path in trx.values():
        df = pd.read_csv(path, index_col=0)
        mindist[i] = np.mean(df["a"])
        i += 1

    mindist = 4 * bl * mindist

    distances = np.zeros((nflies, nflies, m[1]))
    angles = np.zeros((nflies, nflies, m[1]))

    dict_dfs = {}
    for fly_name, fly_path in trx.items():
        df = pd.read_csv(fly_path, index_col=0)
        dict_dfs.update({fly_name: df})

    for i in range(nflies):
        for ii in range(nflies):
            fly1_key = list(trx.keys())[i]
            fly2_key = list(trx.keys())[ii]

            df1 = dict_dfs[fly1_key].copy(deep=True)
            df2 = dict_dfs[fly2_key].copy(deep=True)

            df1_array = df1.to_numpy()
            df2_array = df2.to_numpy()

            distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0]) ** 2 + (df1_array[:, 1] - df2_array[:, 1]) ** 2)
            distances[i, ii, :] = distance  # / (a * 4), 4

            checkang = np.arctan2(df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])
            checkang = checkang * 180 / np.pi

            angle = SL.angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
            angles[i, ii, :] = angle

    ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, m[1])))
    ints[ints < 2] = 0
    ints[ints > 1] = 1

    for i in range(nflies):
        for ii in range(nflies):
            if i == ii:
                ints[i, ii, :] = np.zeros(len(angle))

    idx = np.where(ints != 0)
    r, c, v = idx[0], idx[1], idx[2]

    int_times = np.zeros((nflies * m[1], 1))
    int_ind = 0

    for i in range(nflies):
        for ii in np.setxor1d(np.arange(nflies), i):
            temp = np.intersect1d(np.where(r == i), np.where(c == ii))

            if temp.size != 0:
                potential_ints = np.concatenate(([np.inf], np.diff(v[temp]), [np.inf]))
                nints = np.where(potential_ints > 1)[0]
                durations = np.zeros((len(nints) - 1, 1))

                for ni in range(0, len(nints) - 1):
                    # durations[ni] = np.sum(np.arange(nints[ni], nints[ni]).size) + 1
                    # if np.sum(np.arange(nints[ni], nints[ni + 1] - 1).size) < timecut:
                    #     potential_ints[nints[ni]:nints[ni + 1] - 1] = np.nan
                    # else:
                    #     pass

                    int_times[int_ind] = np.sum(np.array([len(potential_ints[nints[ni] : nints[ni + 1]])]))
                    int_ind += 1

                    if movecut:
                        # int_times[int_ind] = int_times[int_ind] - np.sum(too_slow[r[temp[nints[ni]:nints[ni + 1] - 1]], v[temp[nints[ni]:nints[ni + 1] - 1]] : v[temp[nints[ni] : nints[ni + 1] - 1]])
                        pass

    int_times = int_times[: int_ind - 1] / settings.FPS
    int_times = int_times[int_times != 0]

    # print(f"len int_times: {len(int_times)}")
    return int_times


def rotation(input_XY, center, anti_clockwise_angle):
    """Rotates the input_XY coordinates by a given angle about a center."""

    degree = 1  # for radians use degree=0

    r, c = input_XY.shape

    if input_XY.shape[1] != 2:
        raise ValueError("Not enough columns in coordinates XY")

    r, c = len(center), len([center[0]])
    if (r != 1 and c == 2) or (r == 1 and c != 2):
        raise ValueError('Error in the size of the "center" matrix')

    center_coord = input_XY.copy()
    center_coord[:, 0] = center[0]
    center_coord[:, 1] = center[1]

    anti_clockwise_angle = -1 * anti_clockwise_angle

    if degree == 1:
        anti_clockwise_angle = np.deg2rad(anti_clockwise_angle)

    rotation_matrix = np.array(
        [
            [np.cos(anti_clockwise_angle), -np.sin(anti_clockwise_angle)],
            [np.sin(anti_clockwise_angle), np.cos(anti_clockwise_angle)],
        ]
    )

    rotated_coords = np.dot((input_XY - center_coord), rotation_matrix) + center_coord

    return rotated_coords


def pseudo_fast_flag_interactions(trx, timecut, minang, bl, start, exptime, nflies, fps, movecut):
    # sorted_keys = natsort.natsorted(trx.keys())
    # trx = {k: trx[k] for k in sorted_keys}

    start = round(start * 60 * fps + 1)
    timecut = timecut * fps
    m = [1, 41040]
    nflies = len(trx)

    mindist = np.zeros((nflies, 1))
    i = 0
    for fly_key in trx.keys():
        df = trx[fly_key]
        mindist[i] = np.mean(df["a"])
        i += 1

    mindist = 4 * bl * mindist

    distances = np.zeros((nflies, nflies, m[1]))
    angles = np.zeros((nflies, nflies, m[1]))

    dict_dfs = trx
    for i in range(nflies):
        for ii in range(nflies):
            fly1_key = list(trx.keys())[i]
            fly2_key = list(trx.keys())[ii]

            df1 = dict_dfs[fly1_key].copy(deep=True)
            df2 = dict_dfs[fly2_key].copy(deep=True)

            df1_array = df1.to_numpy()
            df2_array = df2.to_numpy()

            distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0]) ** 2 + (df1_array[:, 1] - df2_array[:, 1]) ** 2)
            distances[i, ii, :] = distance  # / (a * 4), 4

            checkang = np.arctan2(df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])
            checkang = checkang * 180 / np.pi

            angle = SL.angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
            angles[i, ii, :] = angle

    ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, m[1])))
    ints[ints < 2] = 0
    ints[ints > 1] = 1

    for i in range(nflies):
        for ii in range(nflies):
            if i == ii:
                ints[i, ii, :] = np.zeros(len(angle))

    idx = np.where(ints != 0)
    r, c, v = idx[0], idx[1], idx[2]

    int_times = np.zeros((nflies * m[1], 1))
    int_ind = 0

    for i in range(nflies):
        for ii in np.setxor1d(np.arange(nflies), i):
            temp = np.intersect1d(np.where(r == i), np.where(c == ii))

            if temp.size != 0:
                potential_ints = np.concatenate(([np.inf], np.diff(v[temp]), [np.inf]))
                nints = np.where(potential_ints > 1)[0]
                durations = np.zeros((len(nints) - 1, 1))

                for ni in range(0, len(nints) - 1):
                    # durations[ni] = np.sum(np.arange(nints[ni], nints[ni]).size) + 1
                    # if np.sum(np.arange(nints[ni], nints[ni + 1] - 1).size) < timecut:
                    #     potential_ints[nints[ni]:nints[ni + 1] - 1] = np.nan
                    # else:
                    #     pass

                    int_times[int_ind] = np.sum(np.array([len(potential_ints[nints[ni] : nints[ni + 1]])]))
                    int_ind += 1

                    if movecut:
                        # int_times[int_ind] = int_times[int_ind] - np.sum(too_slow[r[temp[nints[ni]:nints[ni + 1] - 1]], v[temp[nints[ni]:nints[ni + 1] - 1]] : v[temp[nints[ni] : nints[ni + 1] - 1]])
                        pass

    int_times = int_times[: int_ind - 1] / settings.FPS
    int_times = int_times[int_times != 0]

    return int_times


def get_trx(normalized_dfs, pxpermm, rand_rot):
    trx = {}
    for fly_key in normalized_dfs:
        fly = normalized_dfs[fly_key]
        if rand_rot:
            rand_rot_value = random.randint(1, 360)
            x = fly["pos x"].to_numpy()
            y = fly["pos y"].to_numpy()
            coords = rotation(np.column_stack((x, y)), [0.5, 0.5], np.random.randint(rand_rot_value))

            x_rot, y_rot = coords[:, 0], coords[:, 1]
            theta = fly["ori"].to_numpy()
            a = fly["a"].to_numpy()
            pxpermm_val = pxpermm[fly_key]

            dict_values = {
                "pos x": x_rot,
                "pos y": y_rot,
                "ori": theta,
                "a": a,
                "pxpermm": pxpermm_val,
            }
            trx.update({fly_key: pd.DataFrame(dict_values)})

    return trx


def calculate_interaction(pi, *args):
    trx, tempangle, tempdistance, start, exptime, nflies, fps = args
    return pseudo_fast_flag_interactions(trx, 0, tempangle, tempdistance, start, exptime, nflies, settings.FPS, 0)


def boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime):
    rand_rot = 1
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}

    list_args = []
    for pi in range(nrand2):
        normalized_dfs, pxpermm = SL.normalize_random_group(pick_random_groups)
        nflies = len(normalized_dfs)
        trx = get_trx(normalized_dfs, pxpermm, rand_rot)
        args = (trx, tempangle, tempdistance, start, exptime, nflies, settings.FPS)

        list_args.append((pi,) + args)

    times = [None] * nrand2
    pool = multiprocessing.Pool()
    times = pool.starmap(calculate_interaction, list_args)

    pool.close()
    pool.join()

    return times


def pseudo_group_space_angle_hist(normalized_dfs, pxpermm):
    """ """
    degree_bins = np.arange(-177.5, 177.6, settings.ANGLE_BIN)
    distance_bins = np.arange(0.125, 99.8751, settings.DISTANCE_BIN)
    total = np.zeros((len(degree_bins) + 1, len(distance_bins) - 1))

    for fly1_key, fly2_key in list(itertools.permutations(normalized_dfs.keys(), 2)):
        df1 = normalized_dfs[fly1_key].copy(deep=True)
        df2 = normalized_dfs[fly2_key].copy(deep=True)

        df1_array = df1.to_numpy()
        df2_array = df2.to_numpy()

        a = np.mean(df1_array[:, 3])
        distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0]) ** 2 + (df1_array[:, 1] - df2_array[:, 1]) ** 2)
        distance = np.round(distance / (a * 4), 4)

        checkang = np.arctan2(df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])
        checkang = checkang * 180 / np.pi

        angle = SL.angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
        angle = np.round(angle)

        if settings.MOVECUT:
            movement = np.sqrt(
                (df1_array[:, 0] - np.roll(df1_array[:, 0], 1)) ** 2
                + (df1_array[:, 1] - np.roll(df1_array[:, 1], 1)) ** 2
            )
            movement[0] = movement[1]
            movement = movement / pxpermm[fly1_key] / settings.FPS

            n, c = np.histogram(movement, bins=np.arange(0, 2.51, 0.01))

            peaks, _ = find_peaks((np.max(n) - n) / np.max(np.max(n) - n), prominence=0.05)
            movecut = 0 if len(peaks) == 0 else c[peaks[0]]

            mask = (distance <= settings.DISTANCE_MAX) & (movement > (movecut * pxpermm[fly1_key] / settings.FPS))

        else:
            mask = distance <= settings.DISTANCE_MAX

        angle = angle[mask]
        distance = distance[mask]

        hist, _, _ = np.histogram2d(
            angle,
            distance,
            bins=(degree_bins, distance_bins),
            range=[[-180, 180], [0, 100.0]],
        )

        hist = hist.T
        temp = np.mean([hist[:, 0], hist[:, -1]], axis=0)
        hist = np.hstack((temp[:, np.newaxis], hist, temp[:, np.newaxis]))

        total += hist.T

    total = np.ceil(total)
    # norm_total = np.ceil((total / np.max(total)) * 256)
    # total = total.T
    return total.T


def process_iteration(pick_random_groups):
    normalized_dfs, pxpermm_dict = SL.normalize_random_group(pick_random_groups)

    N = pseudo_group_space_angle_hist(normalized_dfs, pxpermm_dict)
    return N


def boot_pseudo_fly_space(treatment, temp_ind):
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}

    # superN = []
    # for _ in range(len(pick_random_groups)):
    #     normalized_dfs, pxpermm_dict = normalize_random_group(pick_random_groups)
    #     N = group_space_angle_hist(normalized_dfs, pxpermm_dict)
    #     superN.append(N)
    # N = np.sum(superN, axis=0)

    pool = multiprocessing.Pool()
    superN = pool.map(process_iteration, [pick_random_groups] * 15)

    pool.close()
    pool.join()

    return np.sum(superN, axis=0)


def process_group(group_path):
    trx = fileio.load_files_from_folder(group_path, file_format=".csv")
    nflies = len(trx)
    return fast_flag_interactions(
        trx,
        timecut,
        tempangle,
        tempdistance,
        start,
        exptime,
        nflies,
        settings.FPS,
        0,
    )
