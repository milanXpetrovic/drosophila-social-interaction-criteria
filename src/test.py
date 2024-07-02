#%%
import os

import numpy as np

m = 5000
os.makedirs('./test', exist_ok=True)

for i in range(1000):
    dummy_data = np.random.rand(m, m)
    file_path = f'./test/{i}{i}.npy'
    np.save(file_path, dummy_data)