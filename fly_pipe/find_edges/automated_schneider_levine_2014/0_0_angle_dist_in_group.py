# %%
import os
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


def fast_flag_interactions():
    pass


def get_times():
    pass

# %%


if __name__ == '__main__':

    OUTPUT_PATH = os.path.join(
        "../../../data/find_edges/0_0_angle_dist_in_group/", settings.TREATMENT)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    normalization = json.load(open(settings.NROMALIZATION))
    pxpermm = json.load(open(settings.PXPERMM))
    treatment = fileio.load_multiple_folders(settings.TRACKINGS)

    all_hists = []
    for group_name, group_path in treatment.items():
        print(group_name)

        group = fileio.load_files_from_folder(group_path, file_format='.csv')
        normalized_dfs, pxpermm_group = SL.normalize_group(
            group, normalization, pxpermm, group_name)

        hist = SL.group_space_angle_hist(normalized_dfs, pxpermm_group)
        all_hists.append(hist)

    res = np.sum(all_hists, axis=0)
    res = res.T

    np.save("{}/{}".format(OUTPUT_PATH, "real"), hist)

    with multiprocessing.Pool(processes=6) as pool:
        res = pool.map(
            one_run_random, [(treatment, normalization, pxpermm) for _ in range(500)])

    res = np.sum(res, axis=0)
    res = res.T
    np.save("{}/{}".format(OUTPUT_PATH, "null"), res)

# %%

while np.any(~np.any([angle, distance, time], axis=1)):

    real = np.load(
        "/home/milky/fly-pipe/data/find_edges/0_0_angle_dist_in_group/CSf/real.npy")
    null = np.load(
        "/home/milky/fly-pipe/data/find_edges/0_0_angle_dist_in_group/CSf/null.npy")

    superN = real[:, :80, ]
    pseudo_N = null[:, :80]

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
    print(G)

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
    C[0] = np.arange(0, 20+0.25, 0.25)
    C[1] = np.arange(-180, 181, 5)

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

    angle = []
    distance = []
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

        angle.append(tempangle)
        distance.append(tempdistance)

        # ? Times
        # tempangle se prosljeduje get_times() funkciji

        # temps = s[temp_ind] # indexex of random picked flies
        # tstrain = [None]*temps.shape[0]

        def get_tstrain(ii):
            return get_times(temps[ii].name, tempangle, tempdistance, start, exptime, nflies, np.nan, movecut)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(get_tstrain, range(temps.shape[0]))

        tstrain = list(results)
