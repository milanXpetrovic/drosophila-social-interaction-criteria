# %%
import json
import re

import numpy as np
import pandas as pd
import scipy.io

import src.utils.fileio as fileio

path = "/home/milky/sna/data/trackings/CsCh"
all_groups = fileio.load_multiple_folders(path)

normalization_path = "/home/milky/soc/data/input/pxpermm/CsCh.json"
with open(normalization_path, "r") as json_file:
    pxpermm_group = json.load(json_file)

for group_name, group_path in all_groups.items():
    csv_dict = fileio.load_files_from_folder(group_path)
    structured_arrays = []
    for fly_name, fly_path in csv_dict.items():
        df = pd.read_csv(fly_path)
        FPS = 24
        PXPERMM_GROUP = pxpermm_group[group_name]
        FLY_ID = re.findall(r'\d+', fly_name)[0]
        x = np.array([df["pos x"].to_numpy()])
        y = np.array([df["pos y"].to_numpy()])
        theta = np.array([df["ori"].to_numpy()])
        a = np.array([(df["major axis len"]/4).to_numpy()])
        b = np.array([df["minor axis len"].to_numpy()])
        id = np.array([[np.uint8(FLY_ID)]])
        pxpermm = np.array([[np.float64(PXPERMM_GROUP)]])
        fps = np.array([[np.float64(FPS)]])
        structured_array = np.array([(x, y, theta, a, b, id, pxpermm, fps)],
                                    dtype=[('x', object), ('y', object), ('theta', object), ('a', object), ('b', object), ('id', object), ('pxpermm', object), ('fps', object)])

        structured_arrays.append(structured_array)

    trx_csv = np.array(structured_arrays)
    trx_csv = trx_csv.T

    mat_csv = {
        '__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Thu Jul 06 10:49:32 2023',
        '__version__': '1.0',
        '__globals__': [],
        'trx': trx_csv
    }

    scipy.io.savemat(f"./test/{group_name}.mat", mat_csv)
