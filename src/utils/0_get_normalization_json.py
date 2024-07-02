## This script assumes that youre already preprocessed data and have normalziation file

#%%
import json
import math
import os
import sys

import pandas as pd

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join( os.path.abspath(os.path.join(current_path, '..', '..')), 'src'))

import fileio

import settings

TREATMENT = settings.TREATMENT
ARENA_DIAMETER = settings.ARENA_DIEAMETER_MM

TRACKINGS = os.path.join(settings.RAW_DATA, TREATMENT)
treatment = fileio.load_multiple_folders(TRACKINGS)

SAVE_PATH = os.path.join(settings.INPUT_DIR, TREATMENT, "normalization.json")

SCRIPT_OUTPUT = os.path.join(settings.INPUT_DIR, TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

pxpermm_dict = {}
norm = {}
for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path, file_format='.csv')
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)
        min_x, max_x = min(min_x, df["pos x"].min()), max(max_x, df["pos x"].max())
        min_y, max_y = min(min_y, df["pos y"].min()), max(max_y, df["pos y"].max())

    north = (((max_x - min_x) / 2) + min_x, max_y)
    south = (((max_x - min_x) / 2) + min_x, min_y)
    west = (max_x, ((max_y - min_y) / 2) + min_y)
    east = (min_x, ((max_y - min_y) / 2) + min_y)

    dist_south_north = math.dist(south, north)
    dist_east_west = math.dist(east, west)

    pxpermm = (dist_east_west / ARENA_DIAMETER) + (dist_south_north / ARENA_DIAMETER)
    pxpermm = pxpermm / 2

    pxpermm_dict.update({group_name: pxpermm})

    center_x = (east[0] + west[0]) / 2
    center_y = (north[1] + south[1]) / 2
    radius = dist_south_north / 2

    norm.update({group_name: {
        "min_x": min_x,
        "min_y": min_y,
        "x_px_ratio": dist_east_west / 120,
        "y_px_ratio": dist_south_north / 120,
        "x": center_x,
        "y": center_y,
        "radius": radius
    }})


PXPERMM_SAVE_PATH = os.path.join(SCRIPT_OUTPUT, "pxpermm")
os.makedirs(PXPERMM_SAVE_PATH, exist_ok=True)

with open(os.path.join(PXPERMM_SAVE_PATH, f"{TREATMENT}.json"), 'w') as f:
    json.dump(pxpermm_dict, f, indent=4) 

NORMALIZATION_SAVE_PATH = os.path.join(SCRIPT_OUTPUT, "normalization.json")
with open(NORMALIZATION_SAVE_PATH, 'w') as f:
    json.dump(norm, f, indent=4) 