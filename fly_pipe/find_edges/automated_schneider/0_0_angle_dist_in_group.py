import os
import json
import pandas as pd

from fly_pipe.utils import fileio
from fly_pipe.utils import automated_schneider as ss


POPULATION_NAME = "CSf"

INPUT_PATH = "../../data/raw/" + POPULATION_NAME
OUTPUT_PATH = "../../data/find_edges/0_0_angle_dist_in_group/" + POPULATION_NAME

PXPERMM_PATH = "../../data/pxpermm/" + POPULATION_NAME + ".json"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

population = fileio.load_multiple_folders(INPUT_PATH)
pxpermm = json.load(open(PXPERMM_PATH))

for group_name, group_path in population.items():
    group = fileio.load_files_from_folder(group_path, file_format='.csv')
    result = ss.find_distances_and_angles_in_group(
        group, pxpermm[group_name])
    result.to_csv("{}/{}.csv".format(OUTPUT_PATH, group_name))
