
import matplotlib.pyplot as plt
import numpy as np

path = "./data/times"
times = fileio.load_files_from_folder(path, file_format=".npy")

real = []
pseudo = []

for n, p in times.items():
    data = np.load(p)
    data = np.round(data, 2)
    
    data = data[data <= 5]

    if "_real_" in n:
        real.append(data)

    elif "_pseudo_" in n:
        pseudo.append(data)

real = np.concatenate(real)
pseudo = np.concatenate(pseudo)

print(len(real))
print(len(pseudo))

bins = np.arange(0, 5.1, 0.1)

real_hist, _ = np.histogram(real, bins=bins, density=True)
pseudo_hist, _ = np.histogram(pseudo, bins=bins, density=True)

real_cumulative = np.cumsum(real_hist)
pseudo_cumulative = np.cumsum(pseudo_hist)
real_cumulative = real_cumulative / real_cumulative[-1]
pseudo_cumulative = pseudo_cumulative / pseudo_cumulative[-1]

# # Plot the ratio
# bin_centers = (bins[:-1] + bins[1:]) / 2
# plt.figure(figsize=(10, 5))
# plt.bar(bin_centers, real_cumulative, width=0.1, color='green', alpha=0.7, edgecolor='black')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.bar(bin_centers, pseudo_cumulative, width=0.1, color='green', alpha=0.7, edgecolor='black')
# plt.show()

N= pseudo_cumulative - real_cumulative
plt.figure(figsize=(10, 5))
plt.bar(bin_centers, N, width=0.1, color='green', alpha=0.7, edgecolor='black')
plt.show()