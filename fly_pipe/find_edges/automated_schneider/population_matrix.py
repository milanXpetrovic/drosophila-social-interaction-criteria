
import numpy as np
import pandas as pd

from fly_pipe.utils import fileio

INPUT_PATH = "../../data/find_edges/0_1_group_to_population/"
OUTPUT_PATH = "../../data/find_edges/"

group = fileio.load_files_from_folder(INPUT_PATH, file_format='.csv')

degree_bins = np.array([x for x in range(-180, 181, 5)])
distance_bins = np.array([x*0.25 for x in range(0, 81)])
round_bins = pd.DataFrame(0, index=distance_bins, columns=degree_bins)


round
group by
round
group by

res. to df

for row in total.iterrows():
    _, value = row
    degree = value["angle"]
    distance = value["distance"]
    count = value["counts"]
    degree_bin = np.digitize([degree], degree_bins)[0]-1
    distance_bin = np.digitize([distance], distance_bins)[0]-1
    round_bins.iloc[distance_bin, degree_bin] += count

round_bins.to_csv("{}_matrix.csv".format(OUTPUT_PATH))
