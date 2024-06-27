# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d

# while np.any(~np.any([angle, distance, time], axis=1)):
null = np.load("/home/milky/fly-pipe/data/find_edges/0_0_angle_dist_in_group/CSf/null.npy")
null = null.T
real = np.load("/home/milky/fly-pipe/data/find_edges/0_0_angle_dist_in_group/CSf/real.npy")

superN = real[
    :,
    :80,
]
pseudo_N = null[:, :80]

sum_superN = np.sum(superN)
sum_pseudo_N = np.sum(pseudo_N)

N2 = (superN / sum_superN) - (pseudo_N / sum_pseudo_N)

falloff = np.arange(1, N2.shape[0] + 1).astype(float) ** -1

N2 = N2 * np.tile(falloff, (N2.shape[1], 1)).T

N2[N2 < np.percentile(N2[N2 > 0], 95)] = 0

# Apply Gaussian filter
h = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
N2 = convolve2d(N2, h, mode="same")

labeled_array, num_features = ndimage.label(N2)
bcenter = np.where(labeled_array == 1)[0][-5:]
angle_bin = 5

# Find the index of the first pixel along the second dimension of the connected component with value of -angle_bin*2
acenter1_index = np.where(labeled_array[:, int(-2 / angle_bin)] == 2)[0]
acenter1 = acenter1_index[0] if len(acenter1_index) > 0 else None

# Find the index of the first pixel along the second dimension of the connected component with value of angle_bin*2
acenter2_index = np.where(labeled_array[:, int(2 / angle_bin)] == 2)[0]
acenter2 = acenter2_index[0] if len(acenter2_index) > 0 else None

test = np.zeros_like(N2)
test[bcenter[0] : bcenter[-1], acenter1:acenter2] = 1
G = np.where(test != 0)[0]
print(G)

# Find connected components in N2
labeled_array, num_features = ndimage.label(N2)

# Loop through all connected components
for i in range(1, num_features + 1):
    # Check if the i-th connected component intersects with G
    if np.intersect1d(labeled_array[G], labeled_array[labeled_array == i]).size == 0:
        # If not, set the value of the i-th connected component to zero
        N2[labeled_array == i] = 0

# define the maximum distance
C = {}
C[0] = np.arange(0, 20 + 0.25, 0.25)
C[1] = np.arange(-180, 181, 5)

percentile_value = np.percentile(N2[N2 > 0], 75)
N2[N2 < percentile_value] = 0

CC = ndimage.label(N2)
numPixels = np.array(ndimage.sum(N2, CC[0], range(1, CC[1] + 1)))
idx = np.where(numPixels < 5)[0]
N3 = np.copy(N2)

for i in range(1, CC[1] + 1):
    if not set(CC[0][CC[0] == i]).intersection(set(G)):
        N2[CC[0] == i] = 0

# assuming N2, N3, CC, and idx are already defined as per previous code
a, b = np.where(N2 > 0)
if a.size == 0:
    N2 = np.copy(N3)
    for i in range(len(idx)):
        N2[CC[0] == idx[i]] = 0
    a, b = np.where(N2 > 0)

tempangle = np.max(np.abs(C[1][b]))
tempdistance = C[0][np.argmax(a)]
keepitgoing = 1
n = 15
nrand2 = 500
N2 = superN / n - pseudo_N / nrand2
meanN2 = np.mean(N2)

storeN = np.zeros((len(C[0]) - 1, len(C[1]) - 2))
storeN = storeN.T
nrand1 = 500
distance_bin = 5

angle = []
distance = []
# assuming tempangle, tempdistance, superN, pseudo_N, nrand1, and storeN are already defined as per previous code
if tempangle.size != 0 and tempdistance is not None:
    storeN = storeN + (superN / np.sum(superN) - pseudo_N / np.sum(pseudo_N)) / nrand1

    keepitgoing = True
    while keepitgoing:
        temp = N2[
            np.ix_(
                np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance)[0][0] + 1),
                np.arange(np.where(C[1] == -tempangle)[0][0], np.where(C[1] == tempangle)[0][0] + 1),
            )
        ]
        tempmean = np.mean(temp)
        update = 0
        # assuming N2, C, tempangle, tempdistance, and angle_bin are already defined as per previous code
        tempang = N2[
            np.ix_(
                np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance)[0][0] + 1),
                np.arange(
                    np.where(C[1] == -tempangle - angle_bin)[0][0],
                    np.where(C[1] == tempangle + angle_bin)[0][0] + 1,
                ),
            )
        ]
        tempdist = N2[
            np.ix_(
                np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance + distance_bin)[0][0] + 1),
                np.arange(np.where(C[1] == -tempangle)[0][0], np.where(C[1] == tempangle)[0][0] + 1),
            )
        ]
        tempangdist = N2[
            np.ix_(
                np.arange(np.where(C[0] == 1)[0][0], np.where(C[0] == tempdistance + distance_bin)[0][0] + 1),
                np.arange(
                    np.where(C[1] == -tempangle - angle_bin)[0][0],
                    np.where(C[1] == tempangle + angle_bin)[0][0] + 1,
                ),
            )
        ]

        if np.mean(tempangdist) > np.mean(tempang) and np.mean(tempdist):
            if np.prod(tempangdist.shape) * meanN2 > np.sum(tempang):
                update = 3
        elif np.mean(tempang) > np.mean(tempdist):
            if np.prod(tempang.shape) * meanN2 > np.sum(tempang) and np.mean(tempang) > tempmean:
                update = 1
        else:
            if np.prod(tempang.shape) * meanN2 < np.sum(tempdist) and np.mean(tempdist) > tempmean:
                update = 2

        if update == 1:
            tempangle = tempangle + angle_bin
        elif update == 2:
            tempdistance = tempdistance + distance_bin
        elif update == 3:
            tempangle = tempangle + angle_bin
            tempdistance = tempdistance + distance_bin
        else:
            keepitgoing = 0

        # # Time
        # temps = s[temp_ind]
        # tstrain = [None]*len(temps)

        # for ii in range(len(temps)):
        #     tstrain[ii] = get_times(temps[ii].name, tempangle, tempdistance, start, exptime, nflies, np.nan, movecut)
    angle.append(tempangle)
    distance.append(tempdistance)
