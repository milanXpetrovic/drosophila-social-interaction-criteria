# %%
import json
import os
import sys

import numpy as np
import pandas as pd

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join( os.path.abspath(os.path.join(current_path, '..', '..')), 'src'))

import fileio

import settings

TREATMENT = settings.TREATMENT
normalization = json.load(open(settings.NROMALIZATION))
TRACKINGS_RAW_DATA = os.path.join(settings.RAW_DATA, settings.TREATMENT)

SAVE_PATH = os.path.join(settings.TRACKINGS)

FPS = settings.FPS
START = settings.START
END = settings.END 
END_FRAME = END * FPS * 60

treatment = fileio.load_multiple_folders(TRACKINGS_RAW_DATA)

for group_name, group_path in treatment.items():
    group_nonormalization = normalization[group_name]
    os.makedirs(os.path.join(SAVE_PATH, group_name), exist_ok=True)       
    all_flies_paths = fileio.load_files_from_folder(group_path, file_format=".csv")
    for fly_name, fly_path in all_flies_paths.items():
        df = pd.read_csv(fly_path)
        df.rename(
            columns={"pos x": "pos x", "pos y": "pos y", "ori": "ori", "major axis len": "a", "minor axis len": "b"},
            inplace=True,
        )
        df = df[["pos x", "pos y", "ori", "a", "b"]]
        df = df.iloc[START:END_FRAME, :]
        df = df.fillna(method="ffill")
        df["a"] = df["a"] / 4
        df["pos x"] = df["pos x"].subtract(group_nonormalization.get("min_x"))
        df["pos y"] = df["pos y"].subtract(group_nonormalization.get("min_y"))
        df["pos x"] = df["pos x"] / group_nonormalization.get("x_px_ratio")
        df["pos y"] = df["pos y"] / group_nonormalization.get("y_px_ratio")
        mean_ratio = (group_nonormalization.get("x_px_ratio") + group_nonormalization.get("y_px_ratio")) / 2
        df["a"] = df["a"] / mean_ratio

        save_path = os.path.join(SAVE_PATH, group_name, fly_name.replace('.csv', '.npy')) 
        np.save(save_path, df.to_numpy())
