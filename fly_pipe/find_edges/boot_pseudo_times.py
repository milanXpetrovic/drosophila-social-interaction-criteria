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




# %%
mat = scipy.io.loadmat('ptstrain.mat')
# my_array = mat['ptstrain']
# a0 = my_array[0][0]

strain = 'CSf'
nrand2 = 500
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
