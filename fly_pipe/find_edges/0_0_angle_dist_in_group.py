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

import concurrent.futures

from scipy import ndimage
from scipy.signal import convolve2d

from fly_pipe import settings
from fly_pipe.utils import fileio
import fly_pipe.utils.automated_schneider_levine as SL


def one_run_random(tuple_args):
    treatment, normalization, pxpermm = tuple_args

    random_group = SL.pick_random_group(treatment)
    normalized_dfs, pxpermm = SL.normalize_group(random_group, normalization, pxpermm)
    hist_np = SL.group_space_angle_hist(normalized_dfs, pxpermm)
    return hist_np


def fast_flag_interactions(
    trx, timecut, minang, bl, start, exptime, nflies, fps, movecut
):
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

            distance = np.sqrt(
                (df1_array[:, 0] - df2_array[:, 0]) ** 2
                + (df1_array[:, 1] - df2_array[:, 1]) ** 2
            )
            distances[i, ii, :] = distance  # / (a * 4), 4

            checkang = np.arctan2(
                df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0]
            )
            checkang = checkang * 180 / np.pi

            angle = SL.angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
            angles[i, ii, :] = angle

    ints = np.double(np.abs(angles) < minang) + np.double(
        distances < np.tile(mindist, (nflies, 1, m[1]))
    )
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

                    int_times[int_ind] = np.sum(
                        np.array([len(potential_ints[nints[ni] : nints[ni + 1]])])
                    )
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


def pseudo_fast_flag_interactions(
    trx, timecut, minang, bl, start, exptime, nflies, fps, movecut
):
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

    # for fly_name, fly_path in trx.items():
    #     df = pd.read_csv(fly_path, index_col=0)
    #     dict_dfs.update({fly_name: df})

    dict_dfs = trx
    for i in range(nflies):
        for ii in range(nflies):
            fly1_key = list(trx.keys())[i]
            fly2_key = list(trx.keys())[ii]

            df1 = dict_dfs[fly1_key].copy(deep=True)
            df2 = dict_dfs[fly2_key].copy(deep=True)

            df1_array = df1.to_numpy()
            df2_array = df2.to_numpy()

            distance = np.sqrt(
                (df1_array[:, 0] - df2_array[:, 0]) ** 2
                + (df1_array[:, 1] - df2_array[:, 1]) ** 2
            )
            distances[i, ii, :] = distance  # / (a * 4), 4

            checkang = np.arctan2(
                df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0]
            )
            checkang = checkang * 180 / np.pi

            angle = SL.angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi)
            angles[i, ii, :] = angle

    ints = np.double(np.abs(angles) < minang) + np.double(
        distances < np.tile(mindist, (nflies, 1, m[1]))
    )
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

                    int_times[int_ind] = np.sum(
                        np.array([len(potential_ints[nints[ni] : nints[ni + 1]])])
                    )
                    int_ind += 1

                    if movecut:
                        # int_times[int_ind] = int_times[int_ind] - np.sum(too_slow[r[temp[nints[ni]:nints[ni + 1] - 1]], v[temp[nints[ni]:nints[ni + 1] - 1]] : v[temp[nints[ni] : nints[ni + 1] - 1]])
                        pass

    int_times = int_times[: int_ind - 1] / settings.FPS
    int_times = int_times[int_times != 0]

    return int_times


def calculate_interaction(pi, *args):
    trx, tempangle, tempdistance, start, exptime, nflies, fps = args
    return pseudo_fast_flag_interactions(
        trx, 0, tempangle, tempdistance, start, exptime, nflies, settings.FPS, 0
    )


def boot_pseudo_times(
    treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime
):
    rand_rot = 1
    pick_random_groups = {
        list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind
    }

    normalized_dfs, pxpermm = SL.normalize_random_group(pick_random_groups)
    nflies = len(normalized_dfs)

    trx = {}
    for fly_key in normalized_dfs:
        fly = normalized_dfs[fly_key]
        if rand_rot:
            rand_rot_value = random.randint(1, 360)
            x = fly["pos x"].to_numpy()
            y = fly["pos y"].to_numpy()
            coords = rotation(
                np.column_stack((x, y)), [0.5, 0.5], np.random.randint(rand_rot_value)
            )

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

    times = [None] * nrand2

    pool = multiprocessing.Pool()

    args = (trx, tempangle, tempdistance, start, exptime, nflies, settings.FPS)

    # Parallelize the computation
    times = pool.starmap(calculate_interaction, [(pi,) + args for pi in range(nrand2)])

    pool.close()
    pool.join()

    # for pi in range(nrand2):
    #     times[pi] = pseudo_fast_flag_interactions(
    #         trx, 0, tempangle, tempdistance, start, exptime, nflies, settings.FPS, 0
    # )

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(
    #             pseudo_fast_flag_interactions,
    #             trx,
    #             0,
    #             tempangle,
    #             tempdistance,
    #             start,
    #             exptime,
    #             nflies,
    #             settings.FPS,
    #             0,
    #         )
    #         for _ in range(nrand2)
    #     ]

    # for pi, future in enumerate(concurrent.futures.as_completed(futures)):
    #     times[pi] = future.result()

    return times


def process_iteration(pick_random_groups):
    normalized_dfs, pxpermm_dict = SL.normalize_random_group(pick_random_groups)
    N = SL.group_space_angle_hist(normalized_dfs, pxpermm_dict)
    return N


def boot_pseudo_fly_space(treatment, temp_ind):
    pick_random_groups = {
        list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind
    }

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


warnings.filterwarnings("ignore")

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
start_time = time.time()
while ni < 500:
    print(ni)
    # temp_ind = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)

    temp_ind = [11, 16, 17, 14, 22, 18, 8, 1, 9, 15, 6, 24, 4, 20, 3]
    temp_ind = [x - 1 for x in temp_ind]

    pick_random_groups = {
        list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind
    }

    # all_hists = []
    # for group_name, group_path in pick_random_groups.items():
    #     group = fileio.load_files_from_folder(group_path, file_format=".csv")
    #     normalized_dfs, pxpermm_group = SL.normalize_group(
    #         group, normalization, pxpermm, group_name
    #     )
    #     hist = SL.group_space_angle_hist(normalized_dfs, pxpermm_group)
    #     all_hists.append(hist)

    # res = np.sum(all_hists, axis=0)
    # superN = res

    # np.save("/home/milky/fly-pipe/fly_pipe/find_edges/superN.npy", superN)

    superN = np.load("/home/milky/fly-pipe/fly_pipe/find_edges/superN.npy")
    # superN = superN.T

    pseudo_N = boot_pseudo_fly_space(treatment, temp_ind)
    # print("boot_pseudo_fly_space" + str(time.time() - start_time))

    N2 = (superN / np.sum(superN)) - (pseudo_N / np.sum(pseudo_N))
    falloff = np.arange(1, N2.shape[0] + 1).astype(float) ** -1
    N2 = N2 * np.tile(falloff, (N2.shape[1], 1)).T
    N2[N2 < np.percentile(N2[N2 > 0], 95)] = 0

    # Apply Gaussian filter
    h = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    N2 = convolve2d(N2, h, mode="same")

    labeled_array, num_features = ndimage.label(N2)
    bcenter = np.where(labeled_array == 1)[0][-5:]
    angle_bin = 5

    # Find the index of the first pixel along the second dimension of the connected component with value of -angle_bin*2
    acenter1_index = np.where(labeled_array[:, int(-2 / angle_bin)] == 2)[0]
    acenter1 = acenter1_index[0] if len(acenter1_index) > 0 else None

    # Find the index of the first pixel along the second dimension of the connected component with value of angle_bin*2
    acenter2_index = np.where(labeled_array[:, int(2 / angle_bin)] == 2)[0]
    acenter2 = acenter2_index[0] if len(acenter2_index) > 0 else None

    test = np.zeros_like(N2)
    test[bcenter[0] : bcenter[-1], acenter1:acenter2] = 1
    G = np.where(test != 0)[0]

    # Find connected components in N2
    labeled_array, num_features = ndimage.label(N2)

    # Loop through all connected components
    for i in range(1, num_features + 1):
        # Check if the i-th connected component intersects with G
        if (
            np.intersect1d(labeled_array[G], labeled_array[labeled_array == i]).size
            == 0
        ):
            # If not, set the value of the i-th connected component to zero
            N2[labeled_array == i] = 0
    # define the maximum distance
    C = {}
    C[0] = np.arange(0, settings.DISTANCE_MAX, settings.DISTANCE_BIN_SIZE)
    C[1] = np.arange(-180, 181, settings.DEGREE_BIN_SIZE)

    # percentile_value = np.percentile(N2[N2 > 0], 75)
    non_zero_elements = N2[N2 > 0]
    if len(non_zero_elements) > 0:
        percentile_value = np.percentile(non_zero_elements, 75)
    else:
        percentile_value = 0

    N2[N2 < percentile_value] = 0

    CC = ndimage.label(N2)
    numPixels = np.array(ndimage.sum(N2, CC[0], range(1, CC[1] + 1)))
    idx = np.where(numPixels < 5)[0]
    N3 = np.copy(N2)

    for i in range(1, CC[1] + 1):
        if not set(CC[0][CC[0] == i]).intersection(set(G)):
            N2[CC[0] == i] = 0
    a, b = np.where(N2 > 0)
    if a.size == 0:
        N2 = np.copy(N3)
        for i in range(len(idx)):
            N2[CC[0] == idx[i]] = 0
        a, b = np.where(N2 > 0)

    tempangle = np.max(np.abs(C[1][b]))
    tempdistance = C[0][np.argmax(a)]
    keepitgoing = 1
    n = 15
    nrand2 = 500
    N2 = superN / n - pseudo_N / nrand2
    meanN2 = np.mean(N2)

    nrand1 = 500
    storeN = np.zeros((len(C[0]) - 1, len(C[1]) - 2))
    storeN = storeN.T
    storeT = np.zeros((len(np.arange(0, 30 * 60, 0.05)), nrand1))
    # FIX THIS LEN TO 36000

    distance_bin = 5

    if tempangle.size != 0 and tempdistance is not None:
        storeN = (
            storeN + (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1
        )

        keepitgoing = True
        while keepitgoing:
            temp = N2[
                (C[0] == 1).nonzero()[0][0] : (C[0] == tempdistance).nonzero()[0][0],
                (C[1] >= -tempangle)
                .nonzero()[0][0] : (C[1] == tempangle)
                .nonzero()[0][0]
                + 1,
            ]

            tempmean = temp.mean()
            update = 0

            tempang = N2[
                (C[0] == 1).nonzero()[0][0] : (C[0] == tempdistance).nonzero()[0][0],
                (C[1] >= -tempangle - angle_bin)
                .nonzero()[0][0] : (C[1] == tempangle + angle_bin)
                .nonzero()[0][0]
                + 1,
            ]

            tempdist = N2[
                (C[0] == 1)
                .nonzero()[0][0] : (C[0] == tempdistance + distance_bin)
                .nonzero()[0][0],
                (C[1] >= -tempangle)
                .nonzero()[0][0] : (C[1] == tempangle)
                .nonzero()[0][0]
                + 1,
            ]

            tempangdist = N2[
                (C[0] == 1)
                .nonzero()[0][0] : (C[0] == tempdistance + distance_bin)
                .nonzero()[0][0],
                (C[1] >= -tempangle - angle_bin)
                .nonzero()[0][0] : (C[1] == tempangle + angle_bin)
                .nonzero()[0][0]
                + 1,
            ]

            if np.mean(tempangdist) > np.mean(tempang) and np.mean(tempdist):
                if np.prod(tempangdist.shape) * meanN2 > np.sum(tempang):
                    update = 3
            elif np.mean(tempang) > np.mean(tempdist):
                if (
                    np.prod(tempang.shape) * meanN2 > np.sum(tempang)
                    and np.mean(tempang) > tempmean
                ):
                    update = 1
            else:
                if (
                    np.prod(tempang.shape) * meanN2 < np.sum(tempdist)
                    and np.mean(tempdist) > tempmean
                ):
                    update = 2

            if update == 1:
                tempangle = tempangle + angle_bin
            elif update == 2:
                tempdistance = tempdistance + distance_bin
            elif update == 3:
                tempangle = tempangle + angle_bin
                tempdistance = tempdistance + distance_bin
            else:
                keepitgoing = 0

        angle[ni] = tempangle
        distance[ni] = tempdistance

        pick_random_groups = {
            list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind
        }

        tstrain = [None] * len(pick_random_groups)
        start, timecut, exptime = 0, 0, 30
        start_time = time.time()
        TODO: FIX THIS
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

        print("next part before boot_pseudo_times" + str(time.time() - start_time))
        # TODO: check if works faster if replaced list with numpy array
        start_time = time.time()

        # tempangle = 140
        # tempdistance = 1.250
        print(tempangle, tempdistance)
        ptstrain = boot_pseudo_times(
            treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime
        )
        print(time.time() - start_time)

        M = np.arange(0, 30 * 60 + 0.05, 0.05)
        N = np.zeros((len(ptstrain), len(M) - 1))

        for i in range(len(ptstrain)):
            temp = np.histogram(ptstrain[i], bins=M)[0]
            temps = temp / np.sum(temp)
            temps = np.cumsum(temps[::-1])[::-1]
            N[i, :] = temps

        N[np.isnan(N)] = 0
        PN = np.sum(N, axis=0)
        N = np.zeros((len(tstrain), len(M) - 1))  # (15x36001)

        for i in range(len(tstrain)):
            temp = tstrain[i][:-1]
            temp_hist = np.histogram(temp, bins=M)[0]
            temps = temp_hist / np.sum(temp_hist)
            temps_cumsum = np.cumsum(temps[::-1])[::-1]
            N[i, :] = temps_cumsum

        N[np.isnan(N)] = 0
        N = np.sum(N, axis=0)
        temp = N / n - PN / nrand2
        ftemp = np.argmax(temp[0 : round(len(M) / 2)])
        keepgoing = True

        try:
            keepgoing = True

            while keepgoing:
                curmean = np.mean(temp[0:ftemp])
                posmean = np.mean(temp[0 : ftemp + 1])

                if curmean < posmean:
                    ftemp += 1
                else:
                    keepgoing = False

                if ftemp >= len(temp):
                    ftemp -= 1
                    keepgoing = False

            storeT[:, ni] = temp
            ftemp = np.where(N * 0.5 < N[ftemp])[0]

            if len(ftemp) > 0:
                ftemp = ftemp[0]
                time_arr[ni] = M[ftemp]
                print(
                    str(ni)
                    + " distance "
                    + str(distance[ni])
                    + " angle "
                    + str(angle[ni])
                    + " time "
                    + str(time_arr[ni])
                )

                ni += 1

        except Exception as e:
            print(e)
            # Could not find a good time estimate, so scrap this iteration
            storeN = (
                storeN
                - (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1
            )
            distance[ni] = 0
            angle[ni] = 0
            time[ni] = 0
