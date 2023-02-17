import os
import pandas as pd

from utils import fileio

NAME = "06-01-2023_08-17"
INPUT_PATH = "./data/raw/" + NAME
OUTPUT_PATH = "./data/preproc/0_0_unpack_xlsx/"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

arenas = fileio.load_multiple_folders(INPUT_PATH)

for arena, path in arenas.items():
    if not os.path.exists(OUTPUT_PATH+"/"+arena):
        os.makedirs(OUTPUT_PATH+"/"+NAME+"_"+arena)

    excels = fileio.load_files_from_folder(path, '.xls')

    for xls, path in excels.items():
        xls_file = pd.read_excel(path, sheet_name=None)
        save_location = str(OUTPUT_PATH+"/"+NAME+"_"+arena+"/")
        for sheet_name, df in xls_file.items():
            if sheet_name != "Sheet1":
                df.to_csv(save_location+sheet_name+".csv", index=False)
