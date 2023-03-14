import pandas as pd

from fly_pipe.utils import fileio


POPULATION_NAME = "CSf"
INPUT_PATH = "../../data/find_edges/0_0_angle_dist_in_group/" + POPULATION_NAME
OUTPUT_PATH = "../../data/find_edges/0_1_group_to_population/" + POPULATION_NAME

group = fileio.load_files_from_folder(INPUT_PATH, file_format='.csv')
result = pd.DataFrame()

for name, path in group.items():
    df = pd.read_csv(path, index_col=0)
    df = df.groupby(['angle', 'distance'])[
        'counts'].sum().reset_index(name='counts')
    result = pd.concat([result, df], axis=0)

result = result.groupby(['angle', 'distance'])[
    'counts'].sum().reset_index(name='counts')

result.to_csv("{}.csv".format(OUTPUT_PATH))
