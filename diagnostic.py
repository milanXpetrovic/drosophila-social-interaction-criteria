#%%
"""Quick diagnostic to verify pseudo histogram fix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import interaction_criteria as settings
import src.utils.utils as SL
import src.utils.fileio as fileio
import natsort
import random
import numpy as np
import multiprocessing

# Load data
treatment_data = fileio.load_multiple_folders(settings.TRACKINGS)
sorted_keys = natsort.natsorted(treatment_data.keys())
treatment_data = {k: treatment_data[k] for k in sorted_keys}

print(f"Loaded {len(treatment_data)} groups")

# Sample random indices
random_indices = random.sample(range(len(treatment_data)), settings.RANDOM_GROUP_SIZE)
print(f"Sampled {len(random_indices)} groups")

# Calculate real histogram
treatment_items = list(treatment_data.items())
with multiprocessing.Pool() as pool:
    all_group_hists = pool.starmap(SL.process_norm_group, treatment_items)

super_N = np.sum(all_group_hists, axis=0)
print(f"\nReal histogram sum: {np.sum(super_N):,.0f}")

# Calculate pseudo histogram
pseudo_N = SL.boot_pseudo_fly_space(treatment_data, random_indices)
print(f"Pseudo histogram sum: {np.sum(pseudo_N):,.0f}")

# Check ratio
ratio = np.sum(pseudo_N) / np.sum(super_N)
print(f"\nRatio (Pseudo/Real): {ratio:.2f}")

if 0.3 < ratio < 3.0:
    print("✓ Ratio is reasonable!")
else:
    print(f"✗ PROBLEM: Ratio should be ~0.5-1.5, got {ratio:.2f}")
    print("  Check that num_pseudo_iterations = len(random_group_indices)")
