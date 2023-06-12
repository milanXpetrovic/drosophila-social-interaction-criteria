# %%
import warnings
import itertools
import pandas as pd
import natsort
import random
import os
import sys
import time
import json
import multiprocessing

import numpy as np
from collections import defaultdict

from scipy.ndimage import label
from scipy.signal import convolve2d
from scipy.signal import find_peaks

from skimage import measure as skimage_label

from fly_pipe import settings
from fly_pipe.utils import fileio
import fly_pipe.utils.automated_schneider_levine as SL


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

    normalized_dfs, pxpermm = SL.normalize_random_group(pick_random_groups)
    nflies = len(normalized_dfs)

    list_args = []
    for pi in range(nrand2):
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
    degree_bins = np.arange(-177.5, 177.6, settings.DEGREE_BIN_SIZE)
    distance_bins = np.arange(0.125, 99.8751, settings.DISTANCE_BIN_SIZE)
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


# if __name__ == '__main__':

# OUTPUT_PATH = os.path.join(
#     "../../data/find_edges/0_0_angle_dist_in_group/", settings.TREATMENT)

# os.makedirs(OUTPUT_PATH, exist_ok=True)

# normalization = json.load(open(settings.NROMALIZATION))
# pxpermm = json.load(open(settings.PXPERMM))
# treatment = fileio.load_multiple_folders(settings.TRACKINGS)

# all_hists = []
# for group_name, group_path in treatment.items():
#     print(group_name)

#     group = fileio.load_files_from_folder(group_path, file_format='.csv')
#     normalized_dfs, pxpermm_group = SL.normalize_group(
#         group, normalization, pxpermm, group_name)

#     hist = SL.group_space_angle_hist(normalized_dfs, pxpermm_group)
#     all_hists.append(hist)

# res = np.sum(all_hists, axis=0)
# res = res.T

# np.save("{}/{}".format(OUTPUT_PATH, "real"), hist)

# with multiprocessing.Pool() as pool:
#     res = pool.map(
#         one_run_random, [(treatment, normalization, pxpermm) for _ in range(500)])

# res = np.sum(res, axis=0)
# res = res.T
# np.save("{}/{}".format(OUTPUT_PATH, "null"), res)

# filter out all warnings


# warnings.filterwarnings("ignore")

ni = 0
angle = np.zeros((500, 1))
distance = np.zeros((500, 1))
time_arr = np.zeros((500, 1))

treatment = fileio.load_multiple_folders(settings.TRACKINGS)
normalization = json.load(open(settings.NROMALIZATION))
pxpermm = json.load(open(settings.PXPERMM))

sorted_keys = natsort.natsorted(treatment.keys())
treatment = {k: treatment[k] for k in sorted_keys}

#  np.any(~np.any([angle, distance, time_arr], axis=1))
N2_list = []
angle_bin = 5
while ni < 500:
    ## USE THIS FOR DEBUGG
    # temp_ind = [11, 14, 3, 23, 21, 17, 1, 25, 19, 13, 10, 24, 20, 18, 16]
    # temp_ind = [x - 1 for x in temp_ind]
    # superN = pd.read_csv("/home/milky/fly-pipe/fly_pipe/find_edges/superN.csv", header=None)
    # superN = superN.to_numpy()
    # pseudo_N = pd.read_csv("/home/milky/fly-pipe/fly_pipe/find_edges/pseudoN.csv", header=None)
    # pseudo_N = pseudo_N.to_numpy()

    total_time = time.time()
    temp_ind = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}

    all_hists = []
    for group_name, group_path in pick_random_groups.items():
        group = fileio.load_files_from_folder(group_path, file_format=".csv")
        normalized_dfs, pxpermm_group = SL.normalize_group(group, normalization, pxpermm, group_name)
        hist = SL.group_space_angle_hist(normalized_dfs, pxpermm_group)
        all_hists.append(hist)

    superN = np.sum(all_hists, axis=0)
    pseudo_N = boot_pseudo_fly_space(treatment, temp_ind)

    N2 = (superN / np.sum(superN)) - (pseudo_N / np.sum(pseudo_N))
    falloff = np.arange(1, N2.shape[0] + 1).astype(float) ** -1
    N2 = N2 * np.tile(falloff, (N2.shape[1], 1)).T
    N2[N2 <= np.percentile(N2[N2 > 0], 95)] = 0

    C = {}
    C[0] = np.arange(0, settings.DISTANCE_MAX, settings.DISTANCE_BIN_SIZE)
    C[1] = np.arange(-180, 181, settings.DEGREE_BIN_SIZE)

    a, b = np.where(N2 > 0)
    tempangle = np.max(np.abs(C[1][b - 1]))
    tempdistance = C[0][np.max(a - 1)]

    h = np.array(
        [
            [0.0181, 0.0492, 0.0492, 0.0181],
            [0.0492, 0.1336, 0.1336, 0.0492],
            [0.0492, 0.1336, 0.1336, 0.0492],
            [0.0181, 0.0492, 0.0492, 0.0181],
        ]
    )
    h /= np.sum(h)
    N2 = convolve2d(N2, h, mode="same")

    N2_int = np.where(N2 > 0, 1, N2)
    labeled_image, num_labels = skimage_label.label(N2_int, connectivity=2, return_num=True)
    pixel_idx_list = [np.where(labeled_image == label_num) for label_num in range(1, num_labels + 1)]

    CC = {
        "Connectivity": 8,
        "ImageSize": labeled_image.shape,
        "NumObjects": num_labels,
        "PixelIdxList": pixel_idx_list,
    }

    bcenter = np.where(C[0] < 2)[0][-5:]
    acenter1 = np.where(C[1] == -angle_bin * 2)[0][0]
    acenter2 = np.where(C[1] == angle_bin * 2)[0][0]

    test = np.zeros_like(N2)
    test[bcenter[0] : bcenter[-1] + 1, acenter1 : acenter2 + 1] = 1
    G = np.where(test != 0)

    for i in range(CC["NumObjects"]):
        CC_pixel_idx_list = CC["PixelIdxList"][i]
        CC_set = set(zip(*CC_pixel_idx_list))
        G_set = set(zip(*G))

        if len(CC_set & G_set) == 0:
            N2[pixel_idx_list[i]] = 0

    if not np.any(N2 > 0):
        print("skipping")
        continue

    N2[N2 < np.percentile(N2[N2 > 0], 75)] = 0

    N2_int = np.where(N2 > 0, 1, N2)
    labeled_image, num_labels = skimage_label.label(N2_int, connectivity=2, return_num=True)
    pixel_idx_list = [np.where(labeled_image == label_num) for label_num in range(1, num_labels + 1)]

    CC = {
        "Connectivity": 8,
        "ImageSize": labeled_image.shape,
        "NumObjects": num_labels,
        "PixelIdxList": pixel_idx_list,
    }

    num_pixels = np.array([len(pixel_idx) for pixel_idx in pixel_idx_list])
    idx = np.where(num_pixels < 5)[0]
    N3 = np.copy(N2)

    for i in range(CC["NumObjects"]):
        CC_pixel_idx_list = CC["PixelIdxList"][i]
        CC_set = set(zip(*CC_pixel_idx_list))
        G_set = set(zip(*G))
        intersection = CC_set & G_set

        if len(intersection) == 0:
            N2[pixel_idx_list[i]] = 0

    a, b = np.where(N2 > 0)
    if len(a) == 0:
        N2 = np.copy(N3)
        for i in range(len(idx)):
            N2[CC["PixelIdxList"][idx[i]]] = 0
        a, b = np.where(N2 > 0)

    tempangle = np.max(np.abs(C[1][b]))
    tempdistance = C[0][np.max(a) - 1]
    distance_bin, n = 0.25, 15
    nrand1, nrand2 = 500, 500

    N2 = superN / n - pseudo_N / nrand2
    meanN2 = np.mean(N2)

    storeN = np.zeros((len(C[0]) - 1, len(C[1])))
    storeT = np.zeros((len(np.arange(0, 30 * 60, 0.05)), nrand1))

    keepitgoing = True
    if tempangle.size != 0 and tempdistance is not None:
        storeN = storeN + (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1

        while keepitgoing:
            temp = N2[
                np.where(C[0] == 1)[0][0] : np.where(C[0] == tempdistance)[0][0] + 1,
                np.where(C[1] == -tempangle)[0][0] : np.where(C[1] == tempangle)[0][0] + 1,
            ]

            tempmean = temp.mean()
            update = 0

            tempang = N2[
                np.where((C[0] == 1) | (C[0] == tempdistance))[0][0] : np.where(C[0] == tempdistance)[0][0] + 1,
                np.where((C[1] >= -tempangle - angle_bin) & (C[1] <= tempangle + angle_bin))[0][0] : np.where(
                    (C[1] >= -tempangle - angle_bin) & (C[1] <= tempangle + angle_bin)
                )[0][-1]
                + 1,
            ]
            # tempdist=((N2(find(C{1}==1):find(C{1}==tempdistance+distance_bin),find(C{2}==-tempangle):find(C{2}==tempangle))));
            tempdist = N2[
                np.where((C[0] == 1))[0][0] : np.where((C[0] == tempdistance + distance_bin))[0][0] + 1,
                np.where((C[1] == -tempangle))[0][0] : np.where((C[1] == tempangle))[0][0] + 1,
            ]

            tempangdist = N2[
                np.where((C[0] == 1))[0][0] : np.where((C[0] == tempdistance + distance_bin))[0][0] + 1,
                np.where((C[1] == -tempangle - angle_bin))[0][0] : np.where((C[1] == tempangle + angle_bin))[0][0] + 1,
            ]

            if np.mean(tempangdist) > np.mean(tempang) and np.mean(tempdist):
                if np.prod(tempangdist.shape) * meanN2 > np.sum(tempang):
                    update = 3
                    tempangle = tempangle + angle_bin
                    tempdistance = tempdistance + distance_bin

            elif np.mean(tempang) > np.mean(tempdist):
                if np.prod(tempang.shape) * meanN2 > np.sum(tempang) and np.mean(tempang) > tempmean:
                    update = 1
                    tempangle = tempangle + angle_bin
            else:
                if np.prod(tempang.shape) * meanN2 < np.sum(tempdist) and np.mean(tempdist) > tempmean:
                    update = 2
                    tempdistance = tempdistance + distance_bin

            if update not in [1, 2, 3]:
                keepitgoing = 0

        angle[ni] = tempangle
        distance[ni] = tempdistance

        print(tempangle, tempdistance)

        ## Time
        pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}
        tstrain = [None] * len(pick_random_groups)
        start, timecut, exptime = 0, 0, 30
        start_time = time.time()

        ## TODO: FIX THIS
        for i in range(len(pick_random_groups)):
            key = list(pick_random_groups.keys())[i]
            group_path = pick_random_groups[key]
            trx = fileio.load_files_from_folder(group_path, file_format=".csv")
            nflies = len(trx)

            tstrain[i] = fast_flag_interactions(
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

        # TODO: check if works faster if replaced list with numpy array
        # start_time = time.time()
        ptstrain = boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime)
        # print("boot_pseudo_times", time.time() - start_time)

        M = np.arange(0, 30 * 60 + 0.05, 0.05)
        N = np.zeros((len(ptstrain), len(M) - 1))

        for i in range(len(ptstrain)):
            temp = ptstrain[i][:-1]
            temp = np.histogram(temp, bins=M)[0]
            temps = temp / np.sum(temp)
            temps = np.cumsum(temps[::-1])[::-1]
            N[i, :] = temps

        N[np.isnan(N)] = 0

        PN = np.sum(N, axis=0)
        N = np.zeros((len(tstrain), len(M) - 1))

        for i in range(len(tstrain)):
            temp = tstrain[i][:-1]
            temp = np.histogram(temp, bins=M)[0]
            temps = temp / np.sum(temp)
            temps = np.cumsum(temps[::-1])[::-1]
            N[i, :] = temps

        N[np.isnan(N)] = 0
        N = np.sum(N, axis=0)

        temp = (N / n) - (PN / nrand2)
        ftemp = np.where(temp == np.max(temp[: round(len(M) / 2)]))[0][0]
        keepgoing = True

        try:
            keepgoing = True

            while keepgoing:
                curmean = np.mean(temp[:ftemp])
                posmean = np.mean(temp[: ftemp + 1])

                if curmean < posmean:
                    ftemp += 1
                else:
                    keepgoing = False

                if ftemp >= len(temp):
                    ftemp -= 1
                    keepgoing = False

            storeT[:, ni] = temp
            ftemp_coppy = ftemp
            ftemp = np.where(N * 0.5 < N[ftemp])[0]

            if len(ftemp) > 0 and ftemp[0] != 0:
                ftemp = ftemp[0]
                time_arr[ni] = M[ftemp]
                print(f"{ni} distance {distance[ni]} angle {angle[ni]} time {time_arr[ni]}")
                with open("output.txt", "a") as file:
                    file.write(f"{ni} distance {distance[ni]} angle {angle[ni]} time {time_arr[ni]}")
                    file.write("\n")
                    file.write("TOTAL TIME: " + str(time.time() - total_time))
                    file.write("\n")

                ni += 1

        except Exception as e:
            print(e)
            storeN = storeN - (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1
            distance[ni] = 0
            angle[ni] = 0
            time[ni] = 0
