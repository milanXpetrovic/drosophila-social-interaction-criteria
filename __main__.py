# %%

import json
import multiprocessing
import random
import sys
import time

import natsort
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.signal import convolve2d
from skimage import measure as skimage_label

import src.utils.automated_schneider_levine as SL
import src.utils.fileio as fileio
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
    print(ni)
    total_time = time.time()
    temp_ind = random.sample(range(len(treatment)), settings.RANDOM_GROUP_SIZE)
    pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}

    all_hists = []
    for group_name, group_path in pick_random_groups.items():
        group = fileio.load_files_from_folder(group_path, file_format=".csv")
        normalized_dfs, pxpermm_group = SL.normalize_group(group, normalization, pxpermm, group_name)
        hist = SL.group_space_angle_hist(normalized_dfs, pxpermm_group, is_pseudo=False)
        all_hists.append(hist)

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
    acenter1, acenter2 = np.where(C[1] == -angle_bin * 2)[0][0], np.where(C[1] == angle_bin * 2)[0][0]
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

    if not len(a) or not len(b):
        continue

    tempangle, tempdistance = np.max(np.abs(C[1][b])), C[0][np.max(a)]

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
            tempdist = N2[
                np.where((C[0] == 1))[0][0] : np.where((C[0] == tempdistance + distance_bin))[0][0] + 1,
                np.where((C[1] == -tempangle))[0][0] : np.where((C[1] == tempangle))[0][0] + 1,
            ]
            tempangdist = N2[
                np.where((C[0] == 1))[0][0] : np.where((C[0] == tempdistance + distance_bin))[0][0] + 1,
                np.where((C[1] == -tempangle - angle_bin))[0][0] : np.where((C[1] == tempangle + angle_bin))[0][0]
                + 1,
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

        angle[ni], distance[ni] = tempangle, tempdistance

        ## Time
        pick_random_groups = {list(treatment.keys())[i]: list(treatment.values())[i] for i in temp_ind}
        pool = multiprocessing.Pool()
        args = [(list(pick_random_groups.values())[i], tempangle, tempdistance) for i in range(0, len(temp_ind))]
        results = pool.map(SL.process_group, args)
        tstrain = list(results)
        pool.close()
        pool.join()

        ptstrain = SL.boot_pseudo_times(treatment, nrand2, temp_ind, tempangle, tempdistance, start, exptime)
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
                d = {"distance": distance[ni][0], "angle": angle[ni][0], "time": time_arr[ni][0]}
                d_df = pd.DataFrame([d])
                df = pd.concat([df, d_df], ignore_index=True)
                df.to_csv(f"data/{settings.TREATMENT}_criteria.csv")
                real_array_save = np.concatenate(tstrain)
                pseudo_array_save = np.concatenate(ptstrain)
                np.save(f"data/times/{ni}_real_array", real_array_save)
                np.save(f"data/times/{ni}_pseudo_array", pseudo_array_save)

                with open("data/output.txt", "a") as file:
                    file.write(f"{ni}: {time.time() - total_time} \n")
                
                file.close()
                ni += 1

        except Exception as e:
            print(e)
            storeN = storeN - (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1
            distance[ni] = 0
            angle[ni] = 0
            time_arr[ni] = 0


#%%
import matplotlib.pyplot as plt
import numpy as np

path = "./data/times"
times = fileio.load_files_from_folder(path, file_format=".npy")

real = []
pseudo = []

for n, p in times.items():
    data = np.load(p)
    data = np.round(data, 2)
    
    data = data[data <= 5]

    if "_real_" in n:
        real.append(data)

    elif "_pseudo_" in n:
        pseudo.append(data)

real = np.concatenate(real)
pseudo = np.concatenate(pseudo)

print(len(real))
print(len(pseudo))

bins = np.arange(0, 5.1, 0.1)

real_hist, _ = np.histogram(real, bins=bins, density=True)
pseudo_hist, _ = np.histogram(pseudo, bins=bins, density=True)

real_cumulative = np.cumsum(real_hist)
pseudo_cumulative = np.cumsum(pseudo_hist)
real_cumulative = real_cumulative / real_cumulative[-1]
pseudo_cumulative = pseudo_cumulative / pseudo_cumulative[-1]

# # Plot the ratio
# bin_centers = (bins[:-1] + bins[1:]) / 2
# plt.figure(figsize=(10, 5))
# plt.bar(bin_centers, real_cumulative, width=0.1, color='green', alpha=0.7, edgecolor='black')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.bar(bin_centers, pseudo_cumulative, width=0.1, color='green', alpha=0.7, edgecolor='black')
# plt.show()

N= pseudo_cumulative - real_cumulative
plt.figure(figsize=(10, 5))
plt.bar(bin_centers, N, width=0.1, color='green', alpha=0.7, edgecolor='black')
plt.show()