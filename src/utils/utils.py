import itertools
import json
import multiprocessing
import os
import random
import time

import natsort
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import src.utils.fileio as fileio
from src import settings


def angledifference_nd(angle1, angle2):
    """Calculates the difference between two angles in degrees."""

    diff = angle2 - angle1
    adjustlow, adjusthigh = diff < -180, diff > 180
    while any(adjustlow) or any(adjusthigh):
        diff[adjustlow], diff[adjusthigh] = diff[adjustlow] + 360, diff[adjusthigh] - 360
        adjustlow, adjusthigh = diff < -180, diff > 180

    return diff


def rotation(input_XY, center, anti_clockwise_angle):
    """Rotates the input_XY coordinates by a given angle about a center."""
    degree = 1  # for radians use degree=0
    r, c = input_XY.shape

    if input_XY.shape[1] != 2: raise ValueError("Not enough columns in coordinates XY")
    
    r, c = len(center), len([center[0]])
    if (r != 1 and c == 2) or (r == 1 and c != 2): raise ValueError('Error in the size of the "center" matrix')
    
    center_coord = input_XY.copy()
    center_coord[:, 0], center_coord[:, 1] = center[0], center[1]
    anti_clockwise_angle = -1 * anti_clockwise_angle
    
    if degree == 1: anti_clockwise_angle = np.deg2rad(anti_clockwise_angle)
    rotation_matrix = np.array([[np.cos(anti_clockwise_angle), -np.sin(anti_clockwise_angle)],
                                [np.sin(anti_clockwise_angle), np.cos(anti_clockwise_angle)]])

    return np.dot((input_XY - center_coord), rotation_matrix) + center_coord


def get_trx(normalized_dfs, pxpermm, rand_rot):
    trx = {}
    for fly_key in normalized_dfs:
        fly = normalized_dfs[fly_key]
        # if rand_rot:
        rand_rot_value = random.randint(1, 360)
        x, y = fly["pos x"].to_numpy(), fly["pos y"].to_numpy()
        coords = rotation(np.column_stack((x, y)), [0.5, 0.5], np.random.randint(rand_rot_value))
        dict_values = {
            "pos x": coords[:, 0],
            "pos y": coords[:, 1],
            "ori": fly["ori"].to_numpy(),
            "a": fly["a"].to_numpy(),
            "pxpermm": pxpermm[fly_key],
        }
        trx.update({fly_key: pd.DataFrame(dict_values)})

    return trx


def normalize_group(group, is_pseudo):
    """"""
    normalization = json.load(open(settings.NROMALIZATION))
    pxpermm = json.load(open(settings.PXPERMM))
    normalized_dfs, pxpermm_dict = {}, {}

    if is_pseudo:
        all_flies_paths = {}
        for group_name in random.sample(list(group.keys()), 12):
            norm, group_path = normalization[group_name], group[group_name]
            fly_path = os.path.join(group_path, f"fly{random.randint(0, 11)}.npy")
            
            npy = np.load(fly_path)
            df = pd.DataFrame(npy[:, 1:], columns=["pos x", "pos y", "ori", "a", "b"], index=npy[:, 0])
            df["pos x"] = (df["pos x"] - norm["x"] + norm["radius"]) / (2 * norm["radius"])
            df["pos y"] = (df["pos y"] - norm["y"] + norm["radius"]) / (2 * norm["radius"])
            df["a"] = df["a"] / (2 * norm["radius"])
            normalized_dfs.update({group_name: df})
            pxpermm_dict.update({group_name: pxpermm[group_name] / (2 * norm["radius"])})

    if not is_pseudo:
        for group_name, group_path in group.items():
            norm = normalization[group_name]
            pxpermm_group = pxpermm[group_name] / (2 * norm["radius"])
            all_flies_paths = fileio.load_files_from_folder(group_path, file_format=".npy")
            for fly_name, fly_path in all_flies_paths.items():
                npy = np.load(fly_path)
                df = pd.DataFrame(npy[:, 1:], columns=["pos x", "pos y", "ori", "a", "b"], index=npy[:, 0])
                df["pos x"] = (df["pos x"] - norm["x"] + norm["radius"]) / (2 * norm["radius"])
                df["pos y"] = (df["pos y"] - norm["y"] + norm["radius"]) / (2 * norm["radius"])
                df["a"] = df["a"] / (2 * norm["radius"])
                normalized_dfs.update({fly_name: df})
                pxpermm_dict.update({fly_name: pxpermm_group})
                        
    return normalized_dfs, pxpermm_dict


def group_space_angle_hist(normalized_dfs, pxpermm, is_pseudo):
    """"""
    for fly1_key, fly2_key in list(itertools.permutations(normalized_dfs.keys(), 2)):
        df1, df2 = normalized_dfs[fly1_key].copy(deep=True), normalized_dfs[fly2_key].copy(deep=True)
        df1_array, df2_array = df1.to_numpy(), df2.to_numpy()
        a = np.mean(df1_array[:, 3])
        distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0]) ** 2 + (df1_array[:, 1] - df2_array[:, 1]) ** 2)
        distance = np.round(distance / (a * 4), 4)
        checkang = (np.arctan2(df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])) * 180 / np.pi
        angle = np.round(angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi))

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
        
        else: mask = distance <= settings.DISTANCE_MAX

        degree_bins = np.arange(-177.5, 177.6, settings.ANGLE_BIN)
        distance_bins = np.arange(0.125, 99.8751, settings.DISTANCE_BIN)
        total = np.zeros((len(degree_bins) + 1, len(distance_bins) - 1))

        hist, _, _ = np.histogram2d(angle[mask],
                                    distance[mask],
                                    bins=(degree_bins, distance_bins),
                                    range=[[-180, 180], [0, 100.0]])

        hist = hist.T
        temp = np.mean([hist[:, 0], hist[:, -1]], axis=0)
        hist = np.hstack((temp[:, np.newaxis], hist, temp[:, np.newaxis]))
        total += hist.T

    if is_pseudo:
        total = np.ceil(total)
        return total.T
    
    else:
        norm_total = np.ceil((total / np.max(total)) * 256)
        norm_total = norm_total.T 

        return norm_total


def calculate_N(random_group):
    normalized_dfs, pxpermm_dict = normalize_group(random_group, is_pseudo=True)
    N = group_space_angle_hist(normalized_dfs, pxpermm_dict, is_pseudo=False)
    
    return N


def boot_pseudo_fly_space(treatment, temp_ind):
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}
    
    with multiprocessing.Pool() as pool: superN = pool.map(calculate_N, pick_random_groups)

    N = np.sum(superN, axis=0)

    return N


def fast_flag_interactions(trx, timecut, minang, bl, start, exptime, nflies, fps, movecut):
    """"""
    trx = {k: trx[k] for k in natsort.natsorted(trx.keys())}
    start = round(start * 60 * fps + 1)
    timecut = timecut * fps
    m = [1, 41040]
    nflies = len(trx)

    dict_dfs = {}
    mindist = np.zeros((nflies, 1))
    for i, (fly_name, fly_path) in enumerate(trx.items()):
        npy = np.load(fly_path)
        dict_dfs[fly_name] = pd.DataFrame(npy[:, 1:], columns=["pos x", "pos y", "ori", "a", "b"], index=npy[:, 0])
        mindist[i] = dict_dfs[fly_name]["a"].mean() * 4 * bl

    distances, angles = np.zeros((nflies, nflies, m[1])), np.zeros((nflies, nflies, m[1]))
    for i in range(nflies):
        for ii in range(nflies):
            fly1_key, fly2_key = list(trx.keys())[i], list(trx.keys())[ii]
            df1 = dict_dfs[fly1_key].to_numpy()
            df2 = dict_dfs[fly2_key].to_numpy()
            distance = np.sqrt((df1[:, 0] - df2[:, 0]) ** 2 + (df1[:, 1] - df2[:, 1]) ** 2)
            distances[i, ii, :] = distance 
            checkang = (np.arctan2(df2[:, 1] - df1[:, 1], df2[:, 0] - df1[:, 0])) * 180 / np.pi

            angle = angledifference_nd(checkang, df1[:, 2] * 180 / np.pi)
            angles[i, ii, :] = angle

    ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, m[1])))
    ints[ints < 2] = 0
    ints[ints > 1] = 1

    for i in range(nflies):
        for ii in range(nflies):
            if i == ii:
                ints[i, ii, :] = np.zeros(len(angle))

    r, c, v = np.where(ints != 0)
    int_times = np.zeros((nflies * m[1], 1))
    int_ind = 0

    for i in range(nflies):
        for ii in np.setxor1d(np.arange(nflies), i):
            temp = np.intersect1d(np.where(r == i), np.where(c == ii))

            if temp.size != 0:
                potential_ints = np.concatenate(([np.inf], np.diff(v[temp]), [np.inf]))
                nints = np.where(potential_ints > 1)[0]

                for ni in range(0, len(nints) - 1):
                    int_times[int_ind] = np.sum(np.array([len(potential_ints[nints[ni] : nints[ni + 1]])]))
                    int_ind += 1

    int_times = int_times[: int_ind - 1] / settings.FPS
    int_times = int_times[int_times != 0]

    return int_times
    

def pseudo_fast_flag_interactions(pi, *args):
    trx, minang, bl, start, exptime, nflies, fps = args
    start = round(start * 60 * fps + 1)
    nflies = len(trx)
    total_len = 41040 ## TODO! THIS SHOULD BE DYNAMICALY DETERMINED
    mindist = np.array([[np.mean(trx[fly_key]["a"])] for fly_key in trx.keys()])
    mindist = 4 * bl * mindist

    distances, angles = np.zeros((nflies, nflies, total_len)), np.zeros((nflies, nflies, total_len))
    trx_keys = list(trx.keys())
    
    for i in range(nflies):
        for j in range(nflies):
            fly1_key = trx_keys[i]
            fly2_key = trx_keys[j]
            df1 = trx[fly1_key].to_numpy()
            df2 = trx[fly2_key].to_numpy()
    
            distances[i, j, :] = np.sqrt((df1[:, 0] - df2[:, 0]) ** 2 + (df1[:, 1] - df2[:, 1]) ** 2)
            checkang = (np.arctan2(df2[:, 1] - df1[:, 1], df2[:, 0] - df1[:, 0])) * 180 / np.pi

            angle1 = checkang
            angle2 = df1[:, 2] * 180 / np.pi
            diff = angle2 - angle1
            adjustlow, adjusthigh = diff < -180, diff > 180

            while any(adjustlow) or any(adjusthigh):
                diff[adjustlow], diff[adjusthigh] = diff[adjustlow] + 360, diff[adjusthigh] - 360
                adjustlow, adjusthigh = diff < -180, diff > 180

            angle = diff
            angles[i, j, :] = angle

    ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, total_len)))
    ints[ints < 2], ints[ints > 1] = 0, 1
    ints[np.arange(nflies), np.arange(nflies), :] = 0
    ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, total_len)))
    ints[ints < 2] = 0
    ints[ints > 1] = 1
    
    for i in range(nflies): ints[i, i, :] = np.zeros(trx[list(trx.keys())[0]].shape[0])

    r, c, v = np.where(ints != 0)
    int_times = np.zeros((nflies * total_len, 1))
    int_ind = 0
    for i in range(nflies):
        for j in np.setxor1d(np.arange(nflies), i):          
            temp = np.where((r == i) & (c == j))[0]
            if temp.size != 0:
                potential_ints = np.concatenate(([np.inf], np.diff(v[temp]), [np.inf]))
                durations = np.diff(np.where(potential_ints > 1)[0])
                int_times[int_ind:int_ind+len(durations)] = durations.reshape(-1, 1)
                int_ind += len(durations)

    int_times = int_times[: int_ind - 1] / settings.FPS
    int_times = int_times[int_times != 0]

    return int_times


def boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime):
    rand_rot=1
    times = [None] * nrand2
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}
    args_list = [(pi, pick_random_groups, rand_rot, tempangle, tempdistance, start, exptime, settings.FPS) for pi in range(nrand2)]

    with multiprocessing.Pool() as pool:
        list_args = pool.starmap(normalize_random_group_iteration, args_list)

        ##! HERE
        times = pool.starmap(pseudo_fast_flag_interactions, list_args)

    return times


def normalize_random_group_iteration(pi, pick_random_groups, rand_rot, tempangle, tempdistance, start, exptime, fps):
    normalized_dfs, pxpermm = normalize_group(pick_random_groups, is_pseudo=True)
    trx = get_trx(normalized_dfs, pxpermm, rand_rot)
    nflies=len(normalized_dfs)

    return (pi,) + (trx, tempangle, tempdistance, start, exptime, nflies, fps)


def process_iteration(pick_random_groups):
    normalized_dfs, pxpermm_dict = normalize_group(pick_random_groups, is_pseudo=True)
    N = group_space_angle_hist(normalized_dfs, pxpermm_dict, is_pseudo=True)

    return N


def boot_pseudo_fly_space(treatment, temp_ind):
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}

    with multiprocessing.Pool() as pool: superN = pool.map(process_iteration, [pick_random_groups] * 15)

    return np.sum(superN, axis=0)


def process_norm_group(group_name, group_path):
    group = {group_name: group_path}
    normalized_dfs, pxpermm_group = normalize_group(group, is_pseudo=False)
    hist = group_space_angle_hist(normalized_dfs, pxpermm_group, is_pseudo=False)
    return hist


def process_group(args):
    group_path, tempangle, tempdistance = args
    trx = fileio.load_files_from_folder(group_path, file_format=".npy")
    nflies= len(trx)
    return fast_flag_interactions(
        trx,
        settings.TIMECUT,
        tempangle,
        tempdistance,
        settings.START,
        settings.EXP_DURATION,
        nflies,
        settings.FPS,
        0,
    )
