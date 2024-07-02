# %%

import json
import multiprocessing
import os
import random
import sys
import time

import natsort
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from skimage import measure as skimage_label

import src.utils.fileio as fileio
import src.utils.utils as SL
from src import settings

angle_bin = settings.ANGLE_BIN

distance_bin = settings.DISTANCE_BIN
start = settings.START
timecut = settings.TIMECUT
exptime = settings.EXP_DURATION
n = settings.RANDOM_GROUP_SIZE
nrand1 = settings.N_RANDOM_1
nrand2 = settings.N_RANDOM_2

angle, distance, time_arr = np.zeros((500, 1)), np.zeros((500, 1)), np.zeros((500, 1))

normalization = json.load(open(settings.NROMALIZATION))
pxpermm = json.load(open(settings.PXPERMM))

treatment = fileio.load_multiple_folders(settings.TRACKINGS)
sorted_keys = natsort.natsorted(treatment.keys())
treatment = {k: treatment[k] for k in sorted_keys}

df = pd.DataFrame(columns=["distance", "angle", "time"])
ni = 0

while len(df) < 500:
    # print(ni)
    total_time = time.time()
    temp_ind = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}
    
    treatment_items = treatment.items()
    with multiprocessing.Pool() as pool: all_hists = pool.starmap(SL.process_norm_group, treatment_items)

    superN = np.sum(all_hists, axis=0)
    pseudo_N = SL.boot_pseudo_fly_space(treatment, temp_ind)
    N2 = (superN / np.sum(superN)) - (pseudo_N / np.sum(pseudo_N))
    falloff = np.arange(1, N2.shape[0] + 1).astype(float) ** -1
    N2 = N2 * np.tile(falloff, (N2.shape[1], 1)).T
    N2[N2 <= np.percentile(N2[N2 > 0], 95)] = 0

    C = {}
    C[0] = np.arange(0, settings.DISTANCE_MAX, settings.DISTANCE_BIN)
    C[1] = np.arange(-180, 181, settings.ANGLE_BIN)
    a, b = np.where(N2 > 0)
    tempangle, tempdistance = np.max(np.abs(C[1][b - 1])), C[0][np.max(a - 1)]

    h = np.array([
            [0.0181, 0.0492, 0.0492, 0.0181],[0.0492, 0.1336, 0.1336, 0.0492],
            [0.0492, 0.1336, 0.1336, 0.0492],[0.0181, 0.0492, 0.0492, 0.0181],
        ])
    h /= np.sum(h)

    N2 = convolve2d(N2, h, mode="same")
    N2_int = np.where(N2 > 0, 1, N2)
    labeled_image, num_labels = skimage_label.label(N2_int, connectivity=2, return_num=True)

    CC = {"Connectivity": 8}
    CC["ImageSize "] = labeled_image.shape
    CC["NumObjects"] = num_labels
    CC["PixelIdxList"] = [np.where(labeled_image == label_num) for label_num in range(1, num_labels + 1)]

    bcenter = np.where(C[0] < 2)[0][-5:]
    acenter1, acenter2 = np.where(C[1] == -angle_bin * 2)[0][0], np.where(C[1] == angle_bin * 2)[0][0]
    test = np.zeros_like(N2)
    test[bcenter[0] : bcenter[-1] + 1, acenter1 : acenter2 + 1] = 1
    G = np.where(test != 0)

    for i in range(CC["NumObjects"]):
        CC_pixel_idx_list = CC["PixelIdxList"][i]
        CC_set = set(zip(*CC_pixel_idx_list))
        G_set = set(zip(*G))

        if len(CC_set & G_set) == 0: N2[CC["PixelIdxList"][i]] = 0

    if not np.any(N2 > 0): continue

    N2[N2 < np.percentile(N2[N2 > 0], 75)] = 0
    N2_int = np.where(N2 > 0, 1, N2)
    labeled_image, num_labels = skimage_label.label(N2_int, connectivity=2, return_num=True)

    CC["ImageSize "] = labeled_image.shape
    CC["NumObjects"] = num_labels
    CC["PixelIdxList"] =  [np.where(labeled_image == label_num) for label_num in range(1, num_labels + 1)]

    num_pixels = np.array([len(pixel_idx) for pixel_idx in CC["PixelIdxList"]])
    idx = np.where(num_pixels < 5)[0]
    N3 = np.copy(N2) 

    for i in range(CC["NumObjects"]):
        CC_pixel_idx_list = CC["PixelIdxList"][i]
        CC_set = set(zip(*CC_pixel_idx_list))
        G_set = set(zip(*G))
        intersection = CC_set & G_set

        if len(intersection) == 0: N2[CC["PixelIdxList"][i]] = 0

    a, b = np.where(N2 > 0)

    if len(a) == 0:
        N2 = np.copy(N3)
        for i in range(len(idx)):
            N2[CC["PixelIdxList"][idx[i]]] = 0
        a, b = np.where(N2 > 0)

    if not len(a) or not len(b): continue

    tempangle, tempdistance = np.max(np.abs(C[1][b])), C[0][np.max(a)]
    N2 = superN / n - pseudo_N / nrand2
    meanN2 = np.mean(N2)
    storeN, storeT = np.zeros((len(C[0]) - 1, len(C[1]))), np.zeros((len(np.arange(0, 30 * 60, 0.05)), nrand1))
    
    keepitgoing = True
    if tempangle.size != 0 and tempdistance is not None:
        storeN = storeN + (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1

        while keepitgoing:
            temp = N2[
                np.where(C[0] == 1)[0][0] : np.where(C[0] == tempdistance)[0][0] + 1,
                np.where(C[1] == -tempangle)[0][0] : np.where(C[1] == tempangle)[0][0] + 1
            ]

            tempmean = temp.mean()
            update = 0

            tempang = N2[
                
                np.where((C[0] == 1) | (C[0] == tempdistance))[0][0] : np.where(C[0] == tempdistance)[0][0] + 1,
                np.where((C[1] >= -tempangle - angle_bin) & (C[1] <= tempangle + angle_bin))[0][0] : 
                np.where((C[1] >= -tempangle - angle_bin) & (C[1] <= tempangle + angle_bin))[0][-1] + 1
            ]

            tempdist = N2[
                np.where((C[0] == 1))[0][0] : np.where((C[0] == tempdistance + distance_bin))[0][0] + 1,
                np.where((C[1] == -tempangle))[0][0] : np.where((C[1] == tempangle))[0][0] + 1
            ]

            tempangdist = N2[
                np.where((C[0] == 1))[0][0] : np.where((C[0] == tempdistance + distance_bin))[0][0] + 1,
                np.where((C[1] == -tempangle - angle_bin))[0][0] : np.where((C[1] == tempangle + angle_bin))[0][0] + 1
            ]

            if np.mean(tempangdist) > np.mean(tempang) and np.mean(tempdist):
                if np.prod(tempangdist.shape) * meanN2 > np.sum(tempang):
                    update = 3
                    tempangle += angle_bin
                    tempdistance += distance_bin

            elif np.mean(tempang) > np.mean(tempdist):
                if np.prod(tempang.shape) * meanN2 > np.sum(tempang) and np.mean(tempang) > tempmean:
                    update = 1
                    tempangle += angle_bin

            else: 
                if np.prod(tempang.shape) * meanN2 < np.sum(tempdist) and np.mean(tempdist) > tempmean:
                    update = 2
                    tempdistance += distance_bin

            if update not in [1, 2, 3]: keepitgoing = False

        angle[ni], distance[ni] = tempangle, tempdistance
        
        ## Time
        pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}
        
        args = [(list(pick_random_groups.values())[i], tempangle, tempdistance) for i in range(0, len(temp_ind))]
        
        with multiprocessing.Pool() as pool: tstrain = list(pool.map(SL.process_group, args))

        ptstrain = SL.boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime)

        M = np.arange(0, 30 * 60 + 0.05, 0.05)
        N = np.zeros((len(ptstrain), len(M) - 1))

        for i in range(len(ptstrain)):
            temp = np.histogram(ptstrain[i][:-1], bins=M)[0]
            temps = temp / np.sum(temp)
            temps = np.cumsum(temps[::-1])[::-1]
            N[i, :] = temps

        N[np.isnan(N)] = 0
        PN, N = np.sum(N, axis=0), np.zeros((len(tstrain), len(M) - 1))

        for i in range(len(tstrain)):
            temp = np.histogram(tstrain[i][:-1], bins=M)[0]
            temps = temp / np.sum(temp)
            temps = np.cumsum(temps[::-1])[::-1]
            N[i, :] = temps

        N[np.isnan(N)] = 0
        N = np.sum(N, axis=0)
        temp = (N / n) - (PN / nrand2)
        ftemp = np.where(temp == np.max(temp[: round(len(M) / 2)]))[0][0]

        try:
            keepgoing = True
            while keepgoing:
                if np.mean(temp[:ftemp]) < np.mean(temp[: ftemp + 1]): ftemp += 1
                else: keepgoing = False

                if ftemp >= len(temp):
                    ftemp -= 1
                    keepgoing = False

            storeT[:, ni] = temp
            ftemp = np.where(N * 0.5 < N[ftemp])[0]

            if len(ftemp) > 0 and ftemp[0] != 0:
                time_arr[ni] = M[ftemp[0]]

                print(f"{ni} distance {distance[ni]} angle {angle[ni]} time {time_arr[ni]}")

                d_df = pd.DataFrame([{"distance": distance[ni][0], "angle": angle[ni][0], "time": time_arr[ni][0]}])
                df = pd.concat([df, d_df], ignore_index=True)
                df.to_csv(f"data/{settings.TREATMENT}_criteria.csv")

                times_path = "/srv/milky/drosophila-datasets/CSf_times"
                np.save(os.path.join(times_path, f"{ni}_real_array.npy"),  np.concatenate(tstrain))
                np.save(os.path.join(times_path, f"{ni}_pseudo_array.npy"), np.concatenate(ptstrain))

                with open("data/output.txt", "a") as file: file.write(f"{ni}: {time.time() - total_time} \n")
                file.close()

                ni += 1
 
        except Exception as e:
            print(e)
            storeN = storeN - (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1
            distance[ni], angle[ni], time_arr[ni] = 0, 0, 0
