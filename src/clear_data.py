# %%
import math
import os

import pandas as pd
import toml

import fileio as fileio

FPS = 24

path = "/home/milky/droso-social-interaction-criteria/data/input/trackings/fx"
treatment = fileio.load_multiple_folders(path)

for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path)

    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)

        start = 0
        end = 600 * FPS
        df = df.fillna(method="ffill")
        df = df.loc[start:end]
        # df["a"] = df["a"] * 4
        # df.drop(["Unnamed: 0.2"], axis=1, inplace=True)
        df.to_csv(group_path + "/" + fly_name, index=False)

print(group_path + "/" + fly_name)
