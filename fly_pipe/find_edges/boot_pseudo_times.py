# %%
import sys
import random
import math

import scipy.io
import pandas as pd
import numpy as np

from fly_pipe import settings
from fly_pipe.utils import fileio
import fly_pipe.utils.automated_schneider_levine as SL


def rotation(input_XY, center, anti_clockwise_angle):
    """Rotates the input_XY coordinates by a given angle about a center."""

    degree = 1  # for radians use degree=0

    r, c = input_XY.shape

    if input_XY.shape[1] != 2:
        raise ValueError('Not enough columns in coordinates XY')

    r, c = len(center), len([center[0]])
    if (r != 1 and c == 2) or (r == 1 and c != 2):
        raise ValueError('Error in the size of the "center" matrix')

    center_coord = input_XY.copy()
    center_coord[:, 0] = center[0]
    center_coord[:, 1] = center[1]

    anti_clockwise_angle = -1 * anti_clockwise_angle

    if degree == 1:
        anti_clockwise_angle = np.deg2rad(anti_clockwise_angle)

    rotation_matrix = np.array([[np.cos(anti_clockwise_angle), -np.sin(anti_clockwise_angle)],
                                [np.sin(anti_clockwise_angle), np.cos(anti_clockwise_angle)]])

    rotated_coords = np.dot((input_XY - center_coord),
                            rotation_matrix) + center_coord

    return rotated_coords


def pseudo_fast_flag_interactions(trx, timecut, minang, bl, start, exptime, nflies, fps, movecut):
    # sorted_keys = natsort.natsorted(trx.keys())

    # trx = {k: trx[k] for k in sorted_keys}
    start = round(start*60*fps+1)
    timecut = timecut*fps
    m = [1, 41040]
    nflies = len(trx)

    mindist = np.zeros((nflies, 1))
    i = 0
    for fly_key in trx.keys():
        df = trx[fly_key]
        mindist[i] = np.mean(df["a"])
        i += 1

    mindist = 4*bl*mindist

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

            distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0])**2
                               + (df1_array[:, 1] - df2_array[:, 1])**2)
            distances[i, ii, :] = distance  # / (a * 4), 4

            checkang = np.arctan2(
                df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])
            checkang = checkang * 180 / np.pi

            angle = SL.angledifference_nd(checkang, df1_array[:, 2]*180/np.pi)
            angles[i, ii, :] = angle

    ints = np.double(np.abs(angles) < minang) + \
        np.double(distances < np.tile(mindist, (nflies, 1, m[1])))
    ints[ints < 2] = 0
    ints[ints > 1] = 1

    for i in range(nflies):
        for ii in range(nflies):
            if i == ii:
                ints[i, ii, :] = np.zeros(len(angle))

    idx = np.where(ints != 0)
    r, c, v = idx[0], idx[1], idx[2]

    int_times = np.zeros((nflies*m[1], 1))
    int_ind = 0

    for i in range(nflies):
        for ii in np.setxor1d(np.arange(nflies), i):
            temp = np.intersect1d(np.where(r == i), np.where(c == ii))

            if temp.size != 0:
                potential_ints = np.concatenate(
                    ([np.inf], np.diff(v[temp]), [np.inf]))
                nints = np.where(potential_ints > 1)[0]
                durations = np.zeros((len(nints) - 1, 1))

                for ni in range(0, len(nints) - 1):
                    # durations[ni] = np.sum(np.arange(nints[ni], nints[ni]).size) + 1
                    # if np.sum(np.arange(nints[ni], nints[ni + 1] - 1).size) < timecut:
                    #     potential_ints[nints[ni]:nints[ni + 1] - 1] = np.nan
                    # else:
                    #     pass

                    int_times[int_ind] = np.sum(
                        np.array([len(potential_ints[nints[ni]:nints[ni+1]])]))
                    int_ind += 1

                    if movecut:
                        # int_times[int_ind] = int_times[int_ind] - np.sum(too_slow[r[temp[nints[ni]:nints[ni + 1] - 1]], v[temp[nints[ni]:nints[ni + 1] - 1]] : v[temp[nints[ni] : nints[ni + 1] - 1]])
                        pass

    int_times = int_times[:int_ind-1] / settings.FPS
    int_times = int_times[int_times != 0]

    return int_times


def boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime):

    pick_random_groups = {list(treatment.keys())[i]: list(
        treatment.values())[i] for i in temp_ind}

    normalized_dfs, pxpermm = SL.normalize_random_group(pick_random_groups)
    nflies = len(normalized_dfs)

    times = [None] * nrand2
    for pi in range(nrand2):

        trx = {}
        for fly_key in normalized_dfs:
            fly = normalized_dfs[fly_key]
            if rand_rot:
                rand_rot_value = random.randint(1, 360)
                x = fly['pos x'].to_numpy()
                y = fly['pos y'].to_numpy()
                coords = rotation(np.column_stack((x, y)),
                                  [.5, .5], np.random.randint(rand_rot_value))

                x_rot, y_rot = coords[:, 0], coords[:, 1]
                theta = fly['ori'].to_numpy()
                a = fly['a'].to_numpy()
                pxpermm_val = pxpermm[fly_key]

                dict_values = {"pos x": x_rot, "pos y": y_rot,
                               "ori": theta, "a": a, "pxpermm": pxpermm_val}
                trx.update({fly_key: pd.DataFrame(dict_values)})

        times[pi] = pseudo_fast_flag_interactions(
            trx, 0, tempangle, tempdistance, start, exptime, nflies, settings.FPS, movecut)

    return times


# %%
mat = scipy.io.loadmat('ptstrain.mat')
# my_array = mat['ptstrain']
# a0 = my_array[0][0]

strain = 'CSf'
nrand2 = 50
tempangle = 130
tempdistance = 1.5
start = 0
stop = exptime = 30
offsettime = 0
resample = 1
nnflies = nflies = 12
fps = 22.8
timecut, movecut = 0, 0
rand_rot = 1

treatment = fileio.load_multiple_folders(settings.TRACKINGS)
temp_ind = [22, 6, 3, 16, 11, 7, 17, 14, 8, 5, 21, 25, 26, 19, 15]
temp_ind = [x-1 for x in temp_ind]


ptstrain = boot_pseudo_times(
    treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime)
# %%
