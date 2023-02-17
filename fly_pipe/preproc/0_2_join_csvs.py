import sys
import os
import pandas as pd

from utils import fileio


INPUT_PATH = "./data/preproc/0_0_unpack_xlsx/"
OUTPUT_PATH = "./data/preproc/0_1_join_csvs/"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

files = fileio.load_multiple_folders(INPUT_PATH)

for file_name, file_path in files.items():
    if not os.path.exists(OUTPUT_PATH+"/"+file_name):
        os.makedirs(OUTPUT_PATH+"/"+file_name)

    arenas = fileio.load_multiple_folders(file_path)

    for arena_name, arena_path in arenas.items():
        if not os.path.exists(OUTPUT_PATH+"/"+file_name+"/"+arena_name):
            os.makedirs(OUTPUT_PATH+"/"+file_name+"/"+arena_name)

        videos = fileio.load_multiple_folders(arena_path)


d = {}
for video_name, video_path in videos.items():

    csvs = []
    content = fileio.load_files_from_folder(video_path)

    for csv_name, csv_path in content.items():
        csvs.append((csv_name, pd.read_csv(csv_path)))

    d.update({video_name: csvs})


v1_csvs = d['v1']
v2_csvs = d['v2']

pairs = 0
for csv_name_v2, data_v2 in v2_csvs:
    row_v2 = data_v2.iloc[0].values

    x_v2 = row_v2[0]
    y_v2 = row_v2[1]

    for csv_name_v1, data_v1 in v1_csvs:

        row_v1 = data_v1.iloc[-1].values

        x_v1 = row_v1[0]
        y_v1 = row_v1[1]

        if abs(x_v2-x_v1) < 20 and abs(y_v2 - y_v1) < 20:

            print(x_v1, x_v2)
            print(y_v1, y_v2)
            print(csv_name_v2, csv_name_v1)
            pairs += 1

print(pairs)
