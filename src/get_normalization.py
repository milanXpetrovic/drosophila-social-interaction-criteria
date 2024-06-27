# %%
import os
import pandas as pd
import fileio as fileio
import math
from src import settings


INPUT_DIR = r"/home/milky/droso-social-interaction-criteria/data/input/trackings/CsCh"
SCRIPT_OUTPUT = r"/home/milky/droso-social-interaction-criteria/data/input/trackings/CsCh_ctrax"

os.makedirs(SCRIPT_OUTPUT, exist_ok=True)
treatment = fileio.load_multiple_folders(INPUT_DIR)

ARENA_DIAMETER = 61
TREATMENT = "CsCh"
pxpermm_dict = {}
norm = {}
for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path)
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

print(norm)
print(pxpermm_dict)

for group_name, group_path in treatment.items():
    save_path = os.path.join("/home/milky/droso-social-interaction-criteria/data/input/test", group_name)

    os.makedirs(save_path, exist_ok=True)
    fly_dict = fileio.load_files_from_folder(group_path)
    group_norm = norm[group_name]
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)

        df["pos x"] = df["pos x"].subtract(group_norm["min_x"])
        df["pos y"] = df["pos y"].subtract(group_norm["min_y"])

        df["pos x"] = df["pos x"] / group_norm["x_px_ratio"]
        df["pos y"] = df["pos y"] / group_norm["y_px_ratio"]

        mean_ratio = (group_norm["x_px_ratio"] + group_norm["y_px_ratio"]) / 2
        df["a"] = df["a"] / mean_ratio

        df.to_csv(save_path+"/"+fly_name)
# filename = SCRIPT_OUTPUT + "/wt.json"
# with open(filename, "w") as file:
#     json.dump(d, file)
