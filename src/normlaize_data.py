import os
import pandas as pd
import fileio as fileio

parent_path = r"/home/milky/droso-social-interaction-criteria/data/input/trackings/CsCh"
SCRIPT_OUTPUT = r"/home/milky/droso-social-interaction-criteria/data/input/trackings/CsCh"
treatment = fileio.load_files_from_folder(parent_path)


for group_name, group_path in treatment.items():
    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)

    fly_dict = fileio.load_files_from_folder(group_path)
    min_x, min_y = data_utils.find_group_mins(group_path)

    group_norm_path = os.path.join(settings.NORMALIZATION_DIR, TREATMENT, f"{group_name.replace('.csv', '')}.toml")

    with open(group_norm_path, "r") as group_norm:
        group_norm = toml.load(group_norm)

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
