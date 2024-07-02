#%%
import json
import os
import sys

import pandas as pd

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join( os.path.abspath(os.path.join(current_path, '..', '..')), 'src'))

import fileio

import settings

normalization = json.load(open(settings.NROMALIZATION))
pxpermm = json.load(open(settings.PXPERMM))


treatment = fileio.load_multiple_folders(settings.TRACKINGS)

SCRIPT_OUTPUT = "./test"

for group_name, group_path in treatment.items():
    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)
    norm = normalization[group_name]
    fly_dict = fileio.load_files_from_folder(group_path)

    sys.exit()


    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)
        df = df.iloc[START:END, :]

        df = df.fillna(method="ffill")

        df["pos x"] = df["pos x"].subtract(group_norm.get("min_x"))
        df["pos y"] = df["pos y"].subtract(group_norm.get("min_y"))

        df["pos x"] = df["pos x"] / group_norm.get("x_px_ratio")
        df["pos y"] = df["pos y"] / group_norm.get("y_px_ratio")

        mean_ratio = (group_norm.get("x_px_ratio") + group_norm.get("y_px_ratio")) / 2
        df["major axis len"] = df["major axis len"] / mean_ratio

        # df = data_utils.round_coordinates(df, decimal_places=0)

        df.to_csv(os.path.join(SCRIPT_OUTPUT, group_name, fly_name))
