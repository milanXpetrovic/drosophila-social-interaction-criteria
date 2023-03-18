# %%
# 0_2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fly_pipe.utils import fileio

POP_NAME = "CSf"
INPUT_PATH = "../../../data/find_edges/0_1_group_to_population/"+POP_NAME+".csv"
OUTPUT_PATH = "../../../data/find_edges/0_2_heatmap_matrix/"

# group = fileio.load_files_from_folder(INPUT_PATH, file_format='.csv')

df = pd.read_csv(INPUT_PATH, index_col=0)

# Discretize angle column into bins of size 5
df['angle_bins'] = pd.cut(
    df['angle'], bins=np.arange(-180, 186, 5), include_lowest=True)

# Discretize distance column into bins of size 0.25
df['distance_bins'] = pd.cut(df['distance'], bins=np.arange(
    0, 20.25, 0.25), include_lowest=True)
print(df.head())

counts = df.groupby(['distance_bins', 'angle_bins'])[
    'counts'].sum().unstack()


df = counts.head(30)
n = len(df.columns)
m = len(df)

rad = np.linspace(0, 20.25, m)
a = np.linspace(0, 2 * np.pi, n)
r, th = np.meshgrid(rad, a)
z = df.to_numpy().T
plt.figure(figsize=(10, 10))
plt.subplot(projection="polar")
plt.pcolormesh(th, r, z, cmap='jet')
plt.plot(a, r, ls='none', color='k')
plt.grid()
plt.colorbar()
