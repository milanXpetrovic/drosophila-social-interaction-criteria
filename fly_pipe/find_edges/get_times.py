#%%
import json
import random
from fly_pipe import settings
from fly_pipe.utils import fileio
import sys
import fly_pipe.utils.automated_schneider_levine as SL
import pandas as pd
import numpy as np
import itertools

test = pd.read_csv("test_get_times.csv").to_numpy()

# INPUT
file='CSf_movie_16'
minang=angle=140
bl=1.5
start=0
stop=exptime=30
nnflies=12
fps=22.8
timecut, movecut=0, 0

# FUNCTION GET_TIMES/fast_flag_interactions
# get_times(file,angle,bl,start,stop,nnflies,fps,movecut)
# def fast_flag_interactions(normalized_dfs,timecut,minang,bl,start,exptime,nflies,fps,movecut):
# function [ints, int_times, distance, angles, fps] = fast_flag_interactions(filename,timecut,minang,bl,start,exptime,nnflies,fps,movecut)

treatment = fileio.load_multiple_folders(settings.TRACKINGS)
trx = fileio.load_files_from_folder(treatment[file], file_format='.csv')

import natsort
sorted_keys = natsort.natsorted(trx.keys())

trx = {k: trx[k] for k in sorted_keys}

start=round(start*60*fps+1)
timecut=timecut*fps
m = [1,41040]
nflies = len(trx)

mindist = np.zeros((nflies, 1))
i = 0
for path in trx.values():
    df = pd.read_csv(path, index_col=0)
    mindist[i] = np.mean(df["a"])
    i+=1

mindist = 4*bl*mindist

distances = np.zeros((nflies, nflies, m[1]))
angles = np.zeros((nflies, nflies, m[1]))

for i in range(nflies):
    for ii in range(nflies):
        fly1_key = list(trx.keys())[i]
        fly2_key = list(trx.keys())[ii] 
        df1 = pd.read_csv(trx[fly1_key], index_col=0)
        df2 = pd.read_csv(trx[fly2_key], index_col=0)
        
        df1_array = df1.to_numpy()
        df2_array = df2.to_numpy()

        distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0])**2
                                + (df1_array[:, 1] - df2_array[:, 1])**2)
        distances[i, ii, :] = np.round(distance) #/ (a * 4), 4

        checkang = np.arctan2(
            df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])
        checkang = checkang * 180 / np.pi

        angle = SL.angledifference_nd(checkang, df1_array[:, 2]*180/np.pi)
        angles[i, ii, :] = np.round(angle)


ints = np.double(np.abs(angles) < minang) + np.double(distances < np.tile(mindist, (nflies, 1, m[1])))
ints[ints < 2] = 0
ints[ints > 1] = 1

for i in range(nflies):
    for ii in range(nflies):
        if i == ii:
            ints[i, ii, :]= np.zeros(len(angle)) #np.ones(m[1]) 

# print(np.unique(ints, return_counts=True))
#%%
r, c, v = np.nonzero(ints)
int_times = np.zeros((nflies*m[1], 1))
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
                print(potential_ints[nints[ni]:nints[ni+1]])
                
                int_times[int_ind] = sum(potential_ints[nints[ni]:nints[ni+1]])
                int_ind+=1

                if movecut:
                    #int_times[int_ind] = int_times[int_ind] - np.sum(too_slow[r[temp[nints[ni]:nints[ni + 1] - 1]], v[temp[nints[ni]:nints[ni + 1] - 1]] : v[temp[nints[ni] : nints[ni + 1] - 1]]) 
                    pass

            # inds_nan = np.where(np.isnan(potential_ints))[0]
            # ints[i, ii, v[temp[inds_nan]]] = 0
            # inds_inf = np.where(potential_ints[:-1] > 1)[0]
            # ints[i, ii, v[temp[inds_inf]]] = np.inf

# int_times = int_times[:int_ind-1] / settings.FPS
print(len(int_times))
#%%
print(len(np.unique(int_times)))

#%%
print(len(test), len(int_times))