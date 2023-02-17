# %%
import os
import numpy as np
import pandas as pd

from fly_pipe.utils import fileio


POPULATION_NAME = "CSf"
INPUT_PATH = "../../data/find_edges/0_0_angle_dist_in_group/" + POPULATION_NAME
OUTPUT_PATH = "../../data/find_edges/0_1_group_to_population/" + POPULATION_NAME

group = fileio.load_files_from_folder(INPUT_PATH, file_format='.csv')
total = pd.DataFrame()

for name, path in group.items():
    df = pd.read_csv(path, index_col=0)
    df = df.groupby(['angle', 'distance'])[
        'counts'].sum().reset_index(name='counts')
    total = pd.concat([total, df], axis=0)

total = total.groupby(['angle', 'distance'])[
    'counts'].sum().reset_index(name='counts')

total.to_csv("{}.csv".format(OUTPUT_PATH))

degree_bins = np.array([x for x in range(-180, 181, 5)])
distance_bins = np.array([x*0.25 for x in range(0, 81)])
round_bins = pd.DataFrame(0, index=distance_bins, columns=degree_bins)

for row in total.iterrows():
    _, value = row
    degree = value["angle"]
    distance = value["distance"]
    count = value["counts"]
    degree_bin = np.digitize([degree], degree_bins)[0]-1
    distance_bin = np.digitize([distance], distance_bins)[0]-1
    round_bins.iloc[distance_bin, degree_bin] += count

round_bins.to_csv("{}_matrix.csv".format(OUTPUT_PATH))
