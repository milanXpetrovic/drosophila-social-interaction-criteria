# %%
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

from scipy import ndimage
from scipy.signal import convolve2d

from fly_pipe import settings
from fly_pipe.utils import fileio
import fly_pipe.utils.automated_schneider_levine as SL


def one_run_random(tuple_args):
    treatment, normalization, pxpermm = tuple_args

    random_group = SL.pick_random_group(treatment)
    normalized_dfs, pxpermm = SL.normalize_group(
        random_group, normalization, pxpermm)
    hist_np = SL.group_space_angle_hist(normalized_dfs, pxpermm)
    return hist_np


def fast_flag_interactions(trx, timecut, minang, bl, start, exptime, nflies, fps, movecut):
    sorted_keys = natsort.natsorted(trx.keys())

    trx = {k: trx[k] for k in sorted_keys}
    start = round(start*60*fps+1)
    timecut = timecut*fps
    m = [1, 41040]
    nflies = len(trx)

    mindist = np.zeros((nflies, 1))
    i = 0
    for path in trx.values():
        df = pd.read_csv(path, index_col=0)
        mindist[i] = np.mean(df["a"])
        i += 1

    mindist = 4*bl*mindist

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

    print(f"len int_times: {len(int_times)}")
    return int_times


# %%
if __name__ == '__main__':

    OUTPUT_PATH = os.path.join(
        "../../data/find_edges/0_0_angle_dist_in_group/", settings.TREATMENT)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    normalization = json.load(open(settings.NROMALIZATION))
    pxpermm = json.load(open(settings.PXPERMM))
    treatment = fileio.load_multiple_folders(settings.TRACKINGS)

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

# %%
ni = 0

angle = np.zeros((500, 1))
distance = np.zeros((500, 1))
time = np.zeros((500, 1))

treatment = fileio.load_multiple_folders(settings.TRACKINGS)

while np.any(~np.any([angle, distance, time], axis=1)):
    temp_ind = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)
    temp_ind.sort()

    superN = np.load(
        "/home/mile/fly-pipe/data/find_edges/0_0_angle_dist_in_group/CSf/real.npy")

    pseudo_N = SL.boot_pseudo_fly_space(treatment, temp_ind)

    sum_superN = np.sum(superN)
    sum_pseudo_N = np.sum(pseudo_N)

    N2 = (superN / sum_superN) - (pseudo_N / sum_pseudo_N)

    falloff = np.arange(1, N2.shape[0]+1).astype(float)**-1

    N2 = N2 * np.tile(falloff, (N2.shape[1], 1)).T

    N2[N2 < np.percentile(N2[N2 > 0], 95)] = 0

    # Apply Gaussian filter
    h = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    N2 = convolve2d(N2, h, mode='same')

    labeled_array, num_features = ndimage.label(N2)
    bcenter = np.where(labeled_array == 1)[0][-5:]
    angle_bin = 5

    # Find the index of the first pixel along the second dimension of the connected component with value of -angle_bin*2
    acenter1_index = np.where(labeled_array[:, int(-2/angle_bin)] == 2)[0]
    acenter1 = acenter1_index[0] if len(acenter1_index) > 0 else None

    # Find the index of the first pixel along the second dimension of the connected component with value of angle_bin*2
    acenter2_index = np.where(labeled_array[:, int(2/angle_bin)] == 2)[0]
    acenter2 = acenter2_index[0] if len(acenter2_index) > 0 else None

    test = np.zeros_like(N2)
    test[bcenter[0]:bcenter[-1], acenter1:acenter2] = 1
    G = np.where(test != 0)[0]
    # print(G)

    # Find connected components in N2
    labeled_array, num_features = ndimage.label(N2)

    # Loop through all connected components
    for i in range(1, num_features+1):
        # Check if the i-th connected component intersects with G
        if np.intersect1d(labeled_array[G], labeled_array[labeled_array == i]).size == 0:
            # If not, set the value of the i-th connected component to zero
            N2[labeled_array == i] = 0

    # define the maximum distance
    C = {}
    C[0] = np.arange(0, settings.DISTANCE_MAX, settings.DISTANCE_BIN_SIZE)
    C[1] = np.arange(-180, 181, settings.DEGREE_BIN_SIZE)

    percentile_value = np.percentile(N2[N2 > 0], 75)
    N2[N2 < percentile_value] = 0

    CC = ndimage.label(N2)
    numPixels = np.array(ndimage.sum(N2, CC[0], range(1, CC[1]+1)))
    idx = np.where(numPixels < 5)[0]
    N3 = np.copy(N2)

    for i in range(1, CC[1]+1):
        if not set(CC[0][CC[0] == i]).intersection(set(G)):
            N2[CC[0] == i] = 0

    # assuming N2, N3, CC, and idx are already defined as per previous code
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
    N2 = superN/n - pseudo_N/nrand2
    meanN2 = np.mean(N2)

    storeN = np.zeros((len(C[0])-1, len(C[1])-2))
    storeN = storeN.T
    nrand1 = 500
    distance_bin = 5

    # assuming tempangle, tempdistance, superN, pseudo_N, nrand1, and storeN are already defined as per previous code
    if tempangle.size != 0 and tempdistance is not None:
        storeN = storeN + (superN/np.sum(superN) -
                           pseudo_N/np.sum(pseudo_N))/nrand1

        keepitgoing = True
        while keepitgoing:
            temp = N2[np.ix_(np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance)[0][0]+1),
                             np.arange(np.where(C[1] == -tempangle)[0][0], np.where(C[1] == tempangle)[0][0]+1))]
            tempmean = np.mean(temp)
            update = 0
            # assuming N2, C, tempangle, tempdistance, and angle_bin are already defined as per previous code
            tempang = N2[np.ix_(np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance)[0][0]+1),
                                np.arange(np.where(C[1] == -tempangle-angle_bin)[0][0], np.where(C[1] == tempangle+angle_bin)[0][0]+1))]
            tempdist = N2[np.ix_(np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance+distance_bin)[0][0]+1),
                                 np.arange(np.where(C[1] == -tempangle)[0][0], np.where(C[1] == tempangle)[0][0]+1))]
            tempangdist = N2[np.ix_(np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance+distance_bin)[0][0]+1),
                                    np.arange(np.where(C[1] == -tempangle-angle_bin)[0][0], np.where(C[1] == tempangle+angle_bin)[0][0]+1))]

            if np.mean(tempangdist) > np.mean(tempang) and np.mean(tempdist):
                if np.prod(tempangdist.shape)*meanN2 > np.sum(tempang):
                    update = 3
            elif np.mean(tempang) > np.mean(tempdist):
                if np.prod(tempang.shape)*meanN2 > np.sum(tempang) and np.mean(tempang) > tempmean:
                    update = 1
            else:
                if np.prod(tempang.shape)*meanN2 < np.sum(tempdist) and np.mean(tempdist) > tempmean:
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

        pick_random_groups = {list(treatment.keys())[i]: list(
            treatment.values())[i] for i in temp_ind}

        tstrain = [None] * len(pick_random_groups)

        start = 0
        timecut = 0
        exptime = 30

        for i in range(len(pick_random_groups)):

            key = list(pick_random_groups.keys())[i]

            group_path = pick_random_groups[key]

            trx = fileio.load_files_from_folder(group_path, file_format='.csv')

            nflies = len(trx)

            tstrain[i] = fast_flag_interactions(
                trx, timecut, tempangle, tempdistance, start, exptime, nflies, settings.FPS, 0)


# %%
treatment = fileio.load_multiple_folders(settings.TRACKINGS)

# temp_ind = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)
# temp_ind.sort()

minang = angle = 140
bl = 1.5
start = 0
stop = exptime = 30
nnflies = nflies = 12
fps = 22.8
timecut, movecut = 0, 0

pick_random_groups = {
    'CSf_movie_16': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_16',
    'CSf_movie_10': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_10',
    'CSf_movie_24': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_24',
    'CSf_movie_12': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_12',
    'CSf_movie_04': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_04',
    'CSf_movie_13': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_13',
    'CSf_movie_11': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_11',
    'CSf_movie_14': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_14',
    'CSf_movie_21': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_21',
    'CSf_movie_07': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_07',
    'CSf_movie_23': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_23',
    'CSf_movie_26': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_26',
    'CSf_movie_09': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_09',
    'CSf_movie_15': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_15',
    'CSf_movie_20': '/home/mile/fly-pipe/data/input/trackings/CSf/CSf_movie_20',
}
