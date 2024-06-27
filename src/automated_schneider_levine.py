# %%
import itertools
import json
import multiprocessing
import random

import natsort
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import src.fileio as fileio
from src import settings

# def random_pick_groups(treatment: Dict[str, str]) -> Dict[str, str]:
#     """Randomly picks n groups from treatment."""

#     if len(treatment) < settings.RANDOM_GROUP_SIZE:
#         sys.exit(
#             f"Not enough groups in treatment!\nTrying to pick {settings.RANDOM_GROUP_SIZE} from {len(treatment)} treatments"
#         )

#     picked_groups = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)
#     picked_groups.sort()

#     random_pick = {}
#     for group_i in picked_groups:
#         group_name, group_path = list(treatment.items())[group_i]

#         # group = fileio.load_files_from_folder(group_path, file_format='.csv')
#         # fly_path = random.choice(list(group.keys()))

#         random_pick.update({group_name: group_path})

#     return random_pick


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


def normalize_random_group(pick_random_groups):
    """
    #TODO: write docstring
    """
    normalization = json.load(open(settings.NROMALIZATION))
    pxpermm = json.load(open(settings.PXPERMM))

    normalized_dfs = {}
    pxpermm_dict = {}

    # TODO: fix this number 12 in for loop, dynamicaly determine by number of flies
    for group_name in random.sample(list(pick_random_groups.keys()), 12):
        norm = normalization[group_name]
        group_path = pick_random_groups[group_name]
        group_files = fileio.load_files_from_folder(group_path, file_format=".csv")

        fly_path = random.choice(list(group_files.values()))
        df = pd.read_csv(fly_path, index_col=0)

        df["pos x"] = (df["pos x"] - norm["x"] + norm["radius"]) / (2 * norm["radius"])
        df["pos y"] = (df["pos y"] - norm["y"] + norm["radius"]) / (2 * norm["radius"])
        df["a"] = df["a"] / (2 * norm["radius"])

        normalized_dfs.update({group_name: df})

        pxpermm_group = pxpermm[group_name] / (2 * norm["radius"])
        pxpermm_dict.update({group_name: pxpermm_group})

    return normalized_dfs, pxpermm_dict


def normalize_group(group, normalization, pxpermm, group_name):
    normalized_dfs = {}
    pxpermm_dict = {}

    for fly_name, fly_path in group.items():
        if group_name == "random":
            norm = normalization[fly_name]
            pxpermm_group = pxpermm[fly_name] / (2 * norm["radius"])

        else:
            norm = normalization[group_name]
            pxpermm_group = pxpermm[group_name] / (2 * norm["radius"])

        df = pd.read_csv(fly_path, index_col=0)

        df["pos x"] = (df["pos x"] - norm["x"] + norm["radius"]) / (2 * norm["radius"])
        df["pos y"] = (df["pos y"] - norm["y"] + norm["radius"]) / (2 * norm["radius"])

        df["a"] = df["a"] / (2 * norm["radius"])

        normalized_dfs.update({fly_name: df})
        pxpermm_dict.update({fly_name: pxpermm_group})

    return (normalized_dfs, pxpermm_dict)


def group_space_angle_hist(normalized_dfs, pxpermm):
    """Calculate and return a 2D histogram of the angular and distance differences between pairs of flies based on their positions, using normalized dataframes.

    Args:
    normalized_dfs (Dict[str, pd.DataFrame]): A dictionary containing the normalized dataframes of the flies.
    pxpermm (Dict[str, float]): A dictionary containing the conversion factor from pixels to millimeters for each fly.

    Returns:
    A numpy array representing the normalized 2D histogram of the angular and distance differences between pairs of flies.
    """
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

        angle = angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
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

    norm_total = np.ceil((total / np.max(total)) * 256)
    norm_total = norm_total.T

    return norm_total


def calculate_N(random_group):
    normalized_dfs, pxpermm_dict = normalize_random_group(random_group)
    N = group_space_angle_hist(normalized_dfs, pxpermm_dict)
    return N


def boot_pseudo_fly_space(treatment, temp_ind):
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}


    ## TODO: remove old commennted code
    # superN = []
    # # TODO: add multiprocessing here!
    # for _ in range(len(pick_random_groups)):
    #     normalized_dfs, pxpermm_dict = normalize_random_group(pick_random_groups)
    #     N = group_space_angle_hist(normalized_dfs, pxpermm_dict)
    #     superN.append(N)

    # N = np.sum(superN, axis=0)

    pool = multiprocessing.Pool() 
    superN = pool.map(calculate_N, pick_random_groups)
    pool.close()  
    pool.join()

    N = np.sum(superN, axis=0)

    return N


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

            angle = angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
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

            angle = angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
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


def normalize_random_group_iteration(pi, pick_random_groups, rand_rot, tempangle, tempdistance, start, exptime, fps):
    normalized_dfs, pxpermm = normalize_random_group(pick_random_groups)
    nflies = len(normalized_dfs)
    trx = get_trx(normalized_dfs, pxpermm, rand_rot)
    args = (trx, tempangle, tempdistance, start, exptime, nflies, fps)
    return (pi,) + args


def boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime):
    rand_rot = 1
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}

    pool = multiprocessing.Pool()

    args_list = [(pi, pick_random_groups, rand_rot, tempangle, tempdistance, start, exptime, settings.FPS) for pi in range(nrand2)]
    list_args = pool.starmap(normalize_random_group_iteration, args_list)
    times = [None] * nrand2
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

        angle = angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
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
    normalized_dfs, pxpermm_dict = normalize_random_group(pick_random_groups)

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


def process_group(args):
    group_path, tempangle, tempdistance = args
    trx = fileio.load_files_from_folder(group_path, file_format=".csv")
    nflies = len(trx)
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