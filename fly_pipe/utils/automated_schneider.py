import itertools

import numpy as np
import pandas as pd


def find_distances_and_angles_in_group(group, pxpermm):
    """
    group - dictionary of paths and filenames to .csv files
    group : dict {file_name : file_path}

    Returns dataframe of distance, angle counts
    """

    total = pd.DataFrame()
    group_dfs = {fly: pd.read_csv(path) for fly, path in group.items()}
    combinations = list(itertools.permutations(group_dfs.keys(), 2))

    for fly1, fly2 in combinations:
        df1, df2 = group_dfs[fly1], group_dfs[fly2]

        df = pd.DataFrame()
        df['distance'] = np.sqrt(
            np.square(df1['pos x']-df2['pos x']) + np.square(df1['pos y']-df2['pos y']))
        df['distance'] = df['distance'] / df1.a.mean()
        df['distance'] = df['distance'] / (pxpermm)
        df['distance'] = round(df['distance'], 2)

        df = df[df.distance <= 20]

        df['qx2'] = df1['pos x'] + np.cos(df1['ori']) * (df2['pos x'] -
                                                         df1['pos x']) - np.sin(df1['ori']) * (df2['pos y'] - df1['pos y'])
        df['qy2'] = df1['pos y'] + np.sin(df1['ori']) * (df2['pos x'] -
                                                         df1['pos x']) + np.cos(df1['ori']) * (df2['pos y'] - df1['pos y'])

        df['angle'] = round(np.rad2deg(np.arctan2(
            (df['qy2']-df1['pos y']), (df['qx2']-df1['pos x']))))

        df = df.groupby(
            ['angle', 'distance']).size().reset_index(name='counts')

        total = pd.concat([total, df], axis=0)

    return total


def population_matrix():
    pass