# %%
import time
import natsort
import json
import random
from src import settings
from src.utils import fileio
import sys
import src.utils.automated_schneider_levine as SL
import pandas as pd
import numpy as np
import itertools


start_time = time.time()

test = pd.read_csv("test_get_times.csv").to_numpy()

# INPUT
file = "CSf_movie_16"
minang = angle = 140
bl = 1.5
start = 0
stop = exptime = 30
nnflies = 12
fps = 22.8
timecut, movecut = 0, 0

# FUNCTION GET_TIMES/fast_flag_interactions
# get_times(file,angle,bl,start,stop,nnflies,fps,movecut)
# def fast_flag_interactions(normalized_dfs,timecut,minang,bl,start,exptime,nflies,fps,movecut):
# function [ints, int_times, distance, angles, fps] = fast_flag_interactions(filename,timecut,minang,bl,start,exptime,nnflies,fps,movecut)

treatment = fileio.load_multiple_folders(settings.TRACKINGS)

trx = fileio.load_files_from_folder(treatment[file], file_format=".csv")

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

start_time = time.time()

dict_dfs = {}

for fly_name, fly_path in trx.items():
    df = pd.read_csv(fly_path, index_col=0)

    dict_dfs.update({fly_name: df})

for i in range(nflies):
    for ii in range(nflies):
        fly1_key = list(trx.keys())[i]
        fly2_key = list(trx.keys())[ii]
        # df1 = pd.read_csv(trx[fly1_key], index_col=0)
        # df2 = pd.read_csv(trx[fly2_key], index_col=0)

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

print(f"first double loop time: {time.time()-start_time}")

ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, m[1])))
ints[ints < 2] = 0
ints[ints > 1] = 1

for i in range(nflies):
    for ii in range(nflies):
        if i == ii:
            ints[i, ii, :] = np.zeros(len(angle))  # np.ones(m[1])

idx = np.where(ints != 0)
r, c, v = idx[0], idx[1], idx[2]

int_times = np.zeros((nflies * m[1], 1))
int_ind = 0


start_time = time.time()
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


print(f"double loop time: {time.time()-start_time}")


int_times = int_times[: int_ind - 1] / settings.FPS
int_times = int_times[int_times != 0]

print(len(int_times))
print(len(int_times) == 5799)
