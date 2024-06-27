# %%
import os

import pandas as pd

parent_path = r"/home/milky/droso-social-interaction-criteria/data/input/trackings/CsCh"
destination_path = r"/home/milky/droso-social-interaction-criteria/data/input/trackings/CsCh"
subfolders = [f for f in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, f))]

FPS = 24
END = 10*60

for subfolder in subfolders:
    subfolder_path = os.path.join(parent_path, subfolder)
    output_folder_path = os.path.join(destination_path, subfolder)
    os.makedirs(output_folder_path, exist_ok=True)
    csv_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(".csv")]

    for csv_file in csv_files:
        csv_file_path = os.path.join(subfolder_path, csv_file)
        output_file_path = os.path.join(output_folder_path, csv_file)
        df = pd.read_csv(csv_file_path)
        df.rename(
            columns={"pos x": "pos x", "pos y": "pos y", "ori": "ori", "major axis len": "a", "minor axis len": "b"},
            inplace=True,
        )
        df = df[["pos x", "pos y", "ori", "a", "b"]]

        start = 0
        end = END * FPS
        df = df.fillna(method="ffill")
        df = df.loc[start:end]

        df["a"] = df["a"] / 4

        df.to_csv(output_file_path, index=False)
