# %%
"""
FIXED VERSION of Drosophila Social Interaction Criteria Estimation
Based on Schneider & Levine (2014) Biology Letters

CRITICAL FIXES:
1. Proper normalization in time criteria calculation
2. Better validation of spatial and temporal criteria
3. Improved debugging and logging
4. Fixed circular import issues
5. Added convergence detection
"""

import json
import logging
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from skimage import measure as skimage_label

# Fix imports
current_path = Path(__file__).parent
project_root = current_path.parent
sys.path.insert(0, str(project_root))


from settings import interaction_criteria as settings
# from settings import main_settings

import src.utils.utils as SL
import src.utils.fileio as fileio

# ============================================================================
# CONSTANTS FROM SETTINGS
# ============================================================================
ANGLE_BIN = settings.ANGLE_BIN
DISTANCE_BIN = settings.DISTANCE_BIN
START_TIME = settings.START
EXP_DURATION = settings.EXP_DURATION
RANDOM_GROUP_SIZE = settings.RANDOM_GROUP_SIZE
N_RANDOM_2 = settings.N_RANDOM_2
DISTANCE_MAX = settings.DISTANCE_MAX
MAX_ITERATIONS = 500
CONVERGENCE_WINDOW = 50


# ============================================================================
# SETUP LOGGING
# ============================================================================
def setup_logger(log_dir, log_file_name="runtime_log.txt"):
    """Sets up a logger for runtime messages."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_file_name

    logger_instance = logging.getLogger("runtime_logger")
    logger_instance.setLevel(logging.INFO)

    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger_instance.addHandler(file_handler)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)
    
    return logger_instance


# ============================================================================
# DATA LOADING
# ============================================================================
def load_app_data(trackings_path, normalization_path, pxpermm_path, logger):
    """Loads treatment data, normalization, and pixel-per-mm information."""
    logger.info("Loading treatment data...")
    
    if not Path(trackings_path).exists():
        raise FileNotFoundError(f"Trackings path not found: {trackings_path}")
    
    treatment_data = fileio.load_multiple_folders(trackings_path)
    sorted_keys = natsort.natsorted(treatment_data.keys())
    treatment_data = {k: treatment_data[k] for k in sorted_keys}
    
    logger.info(f"  Loaded {len(treatment_data)} groups")

    # Validate normalization and pxpermm files exist
    for path_name, path in [("Normalization", normalization_path), ("PXPERMM", pxpermm_path)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path_name} file not found: {path}")
    
    with open(normalization_path, 'r') as f:
        normalization_data = json.load(f)
    with open(pxpermm_path, 'r') as f:
        pxpermm_data = json.load(f)
    
    logger.info(f"  Normalization: {len(normalization_data)} groups")
    logger.info(f"  PXPERMM: {len(pxpermm_data)} groups")

    return treatment_data


# ============================================================================
# ITERATION TRACKING
# ============================================================================
def initialize_or_load_criteria_df(save_path, treatment_name, logger_instance):
    """Initializes a new DataFrame for criteria or loads an existing one."""
    save_path = Path(save_path)
    
    if save_path.exists():
        criteria_df = pd.read_csv(save_path, index_col=0)
        start_index = len(criteria_df)
        logger_instance.info(f'{treatment_name} - Resuming from iteration {start_index}')
        logger_instance.info(f'  Current medians: D={criteria_df["distance"].median():.2f}, '
                           f'A={criteria_df["angle"].median():.2f}, '
                           f'T={criteria_df["time"].median():.2f}')
    else:
        criteria_df = pd.DataFrame(columns=["distance", "angle", "time"])
        start_index = 0
        logger_instance.info(f'{treatment_name} - Starting from iteration 0')
    
    return criteria_df, start_index


# ============================================================================
# COORDINATE BINS
# ============================================================================
def get_coordinate_bins():
    """Generates coordinate bins for distance and angle."""
    distance_coords = np.arange(0, DISTANCE_MAX, DISTANCE_BIN)
    angle_coords = np.arange(-180, 181, ANGLE_BIN)
    return {0: distance_coords, 1: angle_coords}


# ============================================================================
# SPATIAL HISTOGRAM CALCULATION
# ============================================================================
def calculate_base_spatial_metrics(treatment_data, random_indices, logger):
    """Calculates histograms for all groups and a pseudo-group."""
    treatment_items = list(treatment_data.items())
    
    logger.info("  Calculating spatial histograms for all groups...")
    with multiprocessing.Pool() as pool:
        all_group_hists = pool.starmap(SL.process_norm_group, treatment_items)

    super_N_hist_total = np.sum(all_group_hists, axis=0)
    
    logger.info("  Generating pseudo (null) spatial distribution...")
    pseudo_N_hist_sample = SL.boot_pseudo_fly_space(treatment_data, random_indices)

    logger.info(f"    Super_N sum: {np.sum(super_N_hist_total):.0f}")
    logger.info(f"    Pseudo_N sum: {np.sum(pseudo_N_hist_sample):.0f}")

    return super_N_hist_total, pseudo_N_hist_sample


# ============================================================================
# DENSITY MAP PROCESSING
# ============================================================================
def calculate_initial_social_density_map(super_N_hist, pseudo_N_hist, logger):
    """Calculates the initial density map for identifying regions."""
    if np.sum(super_N_hist) == 0 or np.sum(pseudo_N_hist) == 0:
        logger.warning("    Empty histograms detected!")
        return np.zeros_like(super_N_hist)

    # Normalize and subtract
    density_map = (super_N_hist / np.sum(super_N_hist)) - (pseudo_N_hist / np.sum(pseudo_N_hist))

    # Apply distance falloff (closer interactions weighted more)
    falloff = np.arange(1, density_map.shape[0] + 1).astype(float) ** -1
    density_map = density_map * np.tile(falloff, (density_map.shape[1], 1)).T

    # Keep only top 5% (3rd quartile was too strict, using 95th percentile)
    positive_density_values = density_map[density_map > 0]
    if len(positive_density_values) > 0:
        threshold = np.percentile(positive_density_values, 95)
        density_map[density_map <= threshold] = 0
        logger.info(f"    Positive pixels after threshold: {np.sum(density_map > 0)}")
    else:
        density_map[:] = 0
        logger.warning("    No positive density values!")

    return density_map


def filter_map_by_central_region(target_map, labeled_components, coord_bins):
    """Filters a map, keeping only components overlapping a defined central region."""
    distance_coords = coord_bins[0]
    angle_coords = coord_bins[1]

    bcenter_candidates = np.where(distance_coords < 2)[0]
    if not bcenter_candidates.size:
        return target_map
    bcenter = bcenter_candidates[-5:] if len(bcenter_candidates) >= 5 else bcenter_candidates

    acenter1_idx = np.where(angle_coords == -ANGLE_BIN * 2)[0]
    acenter2_idx = np.where(angle_coords == ANGLE_BIN * 2)[0]

    if not (acenter1_idx.size and acenter2_idx.size and bcenter.size):
        return target_map

    central_mask = np.zeros_like(target_map)
    central_mask[bcenter[0]: bcenter[-1] + 1,
                 acenter1_idx[0]: acenter2_idx[0] + 1] = 1
    central_region_pixels_G = np.where(central_mask != 0)
    central_region_set = set(zip(*central_region_pixels_G))

    filtered_target_map = np.copy(target_map)
    for i in range(labeled_components["NumObjects"]):
        component_pixels = labeled_components["PixelIdxList"][i]
        component_set = set(zip(*component_pixels))
        if not (component_set & central_region_set):
            filtered_target_map[component_pixels] = 0
    
    return filtered_target_map


def process_spatial_density_map(density_map, coord_bins, logger):
    """Applies convolution, labeling, and filtering to the density map."""
    if not np.any(density_map > 0):
        logger.warning("    Density map is empty!")
        return None

    # Gaussian-like smoothing kernel
    h_kernel = np.array([
        [0.0181, 0.0492, 0.0492, 0.0181],
        [0.0492, 0.1336, 0.1336, 0.0492],
        [0.0492, 0.1336, 0.1336, 0.0492],
        [0.0181, 0.0492, 0.0492, 0.0181],
    ])
    h_kernel /= np.sum(h_kernel)
    processed_map = convolve2d(density_map, h_kernel, mode="same")

    # First labeling
    binary_map_1 = np.where(processed_map > 0, 1, 0)
    labeled_image_1, num_labels_1 = skimage_label.label(binary_map_1, connectivity=2, return_num=True)
    
    if num_labels_1 == 0:
        logger.warning("    No connected components found after first labeling!")
        return None

    cc_1 = {
        "NumObjects": num_labels_1,
        "PixelIdxList": [np.where(labeled_image_1 == i) for i in range(1, num_labels_1 + 1)]
    }

    # Filter by central region
    map_after_central_filter_1 = filter_map_by_central_region(processed_map, cc_1, coord_bins)
    
    if not np.any(map_after_central_filter_1 > 0):
        logger.warning("    No pixels remain after central region filter!")
        return None

    # Keep top 25% (75th percentile)
    positive_values = map_after_central_filter_1[map_after_central_filter_1 > 0]
    if not positive_values.size:
        return None

    map_after_central_filter_1[map_after_central_filter_1 < np.percentile(positive_values, 75)] = 0
    
    if not np.any(map_after_central_filter_1 > 0):
        return None

    # Second labeling
    binary_map_2 = np.where(map_after_central_filter_1 > 0, 1, 0)
    labeled_image_2, num_labels_2 = skimage_label.label(binary_map_2, connectivity=2, return_num=True)
    
    if num_labels_2 == 0:
        return None

    cc_2 = {
        "NumObjects": num_labels_2,
        "PixelIdxList": [np.where(labeled_image_2 == i) for i in range(1, num_labels_2 + 1)]
    }

    # Final filtering
    n2_snapshot = np.copy(map_after_central_filter_1)
    final_map_attempt = filter_map_by_central_region(n2_snapshot, cc_2, coord_bins)

    if not np.any(final_map_attempt > 0):
        # Remove small components (< 5 pixels)
        final_map_attempt = np.copy(n2_snapshot)
        for i in range(cc_2["NumObjects"]):
            if len(cc_2["PixelIdxList"][i][0]) < 5:
                final_map_attempt[cc_2["PixelIdxList"][i]] = 0

    if not np.any(final_map_attempt > 0):
        logger.warning("    Final map is empty after all filtering!")
        return None

    logger.info(f"    Final active pixels: {np.sum(final_map_attempt > 0)}")
    return final_map_attempt


# ============================================================================
# ANGLE & DISTANCE DETERMINATION
# ============================================================================
def determine_max_angle_distance_from_map(processed_map, coord_bins):
    """Determines maximum angle and distance from active regions in a map."""
    if processed_map is None or not np.any(processed_map > 0):
        return None, None

    active_rows, active_cols = np.where(processed_map > 0)
    if not active_rows.size or not active_cols.size:
        return None, None

    max_abs_angle = np.max(np.abs(coord_bins[1][active_cols]))
    max_distance = coord_bins[0][np.max(active_rows)]

    return max_abs_angle, max_distance


def refine_angle_distance_iteratively(density_map_for_refinement, coord_bins, 
                                      initial_angle, initial_distance, logger):
    """Iteratively refines angle and distance based on mean densities in expanding regions."""
    mean_overall_density = np.mean(density_map_for_refinement)
    distance_coords = coord_bins[0]
    angle_coords = coord_bins[1]

    refined_angle = float(initial_angle)
    refined_distance = float(initial_distance)

    logger.info(f"    Initial: angle={refined_angle:.1f}°, distance={refined_distance:.2f} BL")

    keep_iterating = True
    iteration_count = 0
    max_refinement_iterations = 100
    
    while keep_iterating and iteration_count < max_refinement_iterations:
        iteration_count += 1
        
        try:
            dist_start_idx = np.where(distance_coords == 1)[0]
            if not dist_start_idx.size:
                break
            dist_start_idx = dist_start_idx[0]

            dist_end_idx = np.where(distance_coords == refined_distance)[0][0] + 1
            ang_start_idx = np.where(angle_coords == -refined_angle)[0][0]
            ang_end_idx = np.where(angle_coords == refined_angle)[0][0] + 1

            current_region_slice = density_map_for_refinement[dist_start_idx:dist_end_idx, 
                                                              ang_start_idx:ang_end_idx]
            if current_region_slice.size == 0:
                break
            mean_current_region = current_region_slice.mean()

            # Check expansions
            ang_expanded_coords_idx = np.where((angle_coords >= -refined_angle - ANGLE_BIN)
                                               & (angle_coords <= refined_angle + ANGLE_BIN))[0]
            dist_expanded_coord_idx = np.where(distance_coords == refined_distance + DISTANCE_BIN)[0]

            if not (ang_expanded_coords_idx.size and dist_expanded_coord_idx.size):
                break

            ang_exp_start_idx, ang_exp_end_idx = ang_expanded_coords_idx[0], ang_expanded_coords_idx[-1] + 1
            dist_exp_end_idx = dist_expanded_coord_idx[0] + 1

            angle_expanded_slice = density_map_for_refinement[dist_start_idx:dist_end_idx,
                                                              ang_exp_start_idx:ang_exp_end_idx]
            distance_expanded_slice = density_map_for_refinement[dist_start_idx:dist_exp_end_idx, 
                                                                 ang_start_idx:ang_end_idx]
            both_expanded_slice = density_map_for_refinement[dist_start_idx:dist_exp_end_idx, 
                                                             ang_exp_start_idx:ang_exp_end_idx]

            if (angle_expanded_slice.size == 0 or distance_expanded_slice.size == 0 or 
                both_expanded_slice.size == 0):
                break

            mean_angle_expanded = angle_expanded_slice.mean()
            mean_distance_expanded = distance_expanded_slice.mean()
            mean_both_expanded = both_expanded_slice.mean()

            # Decide which expansion to apply
            update_type = 0
            if mean_both_expanded > mean_angle_expanded and mean_both_expanded > mean_distance_expanded:
                if (both_expanded_slice.size * mean_overall_density) > np.sum(angle_expanded_slice):
                    update_type = 3  # Expand both
            elif mean_angle_expanded > mean_distance_expanded:
                if ((angle_expanded_slice.size * mean_overall_density > np.sum(angle_expanded_slice)) and 
                    (mean_angle_expanded > mean_current_region)):
                    update_type = 1  # Expand angle
            else:
                if ((distance_expanded_slice.size * mean_overall_density < np.sum(distance_expanded_slice)) and 
                    (mean_distance_expanded > mean_current_region)):
                    update_type = 2  # Expand distance

            # Apply update
            if update_type == 3:
                refined_angle += ANGLE_BIN
                refined_distance += DISTANCE_BIN
            elif update_type == 1:
                refined_angle += ANGLE_BIN
            elif update_type == 2:
                refined_distance += DISTANCE_BIN
            else:
                keep_iterating = False

            # Safety bounds
            if refined_angle > 180:
                refined_angle = 180.0
                keep_iterating = False
            if refined_distance >= DISTANCE_MAX:
                refined_distance = float(DISTANCE_MAX - DISTANCE_BIN)
                keep_iterating = False
            if refined_distance < 0:
                refined_distance = 0.0
                keep_iterating = False

        except (IndexError, ValueError) as e:
            logger.warning(f"    Refinement error: {e}")
            keep_iterating = False

    logger.info(f"    Refined ({iteration_count} steps): angle={refined_angle:.1f}°, "
               f"distance={refined_distance:.2f} BL")

    return refined_angle, refined_distance


# ============================================================================
# TIME DISTRIBUTION CALCULATION - FIXED VERSION
# ============================================================================
def calculate_interaction_time_distributions(treatment_data, random_indices, 
                                             selected_angle, selected_distance, logger):
    """
    Calculates real and pseudo cumulative time distributions for interactions.
    
    CRITICAL FIX: Properly track n_groups and n_pseudo for normalization.
    """
    picked_groups_values = [list(treatment_data.values())[i] for i in random_indices]
    n_groups = len(picked_groups_values)
    
    logger.info(f"  Calculating interaction times: angle={selected_angle:.1f}°, "
               f"distance={selected_distance:.2f} BL")
    
    # Prepare arguments for real group processing
    pool_args_real = [(group_data, selected_angle, selected_distance) 
                      for group_data in picked_groups_values]

    # Calculate real times using multiprocessing
    with multiprocessing.Pool() as pool:
        real_times_per_group = list(pool.map(SL.process_group, pool_args_real))

    # Calculate pseudo times
    logger.info(f"    Generating {N_RANDOM_2} pseudo iterations...")
    pseudo_times_per_group = SL.boot_pseudo_times(
        treatment_data, N_RANDOM_2, random_indices, 
        selected_angle, selected_distance, 
        START_TIME, EXP_DURATION
    )

    # Time bins: 0 to 30 minutes in 0.05s steps
    time_bins = np.arange(0, 30 * 60 + 0.05, 0.05)

    def get_summed_cumulative_histogram(time_series_list, bins, label):
        """Calculate summed reverse cumulative histogram."""
        num_series = len(time_series_list)
        hist_matrix = np.zeros((num_series, len(bins) - 1))
        valid_count = 0
        
        for i, series_data in enumerate(time_series_list):
            if series_data is None or len(series_data) <= 1:
                continue

            # Remove potential artifact (last element)
            if len(series_data) > 1:
                series_data = series_data[:-1]
            
            if len(series_data) == 0:
                continue

            # Histogram of interaction durations
            hist_counts, _ = np.histogram(series_data, bins=bins)
            
            if np.sum(hist_counts) == 0:
                continue

            # CRITICAL: Normalize each group's histogram BEFORE cumsum
            normalized_hist = hist_counts / np.sum(hist_counts)
            
            # Reverse cumulative sum (P(duration >= t))
            cumulative_sum_rev = np.cumsum(normalized_hist[::-1])[::-1]
            
            hist_matrix[i, :] = cumulative_sum_rev
            valid_count += 1

        # Replace NaN with 0
        hist_matrix[np.isnan(hist_matrix)] = 0
        
        # Sum across all series
        summed = np.sum(hist_matrix, axis=0)
        
        logger.info(f"    {label}: {valid_count}/{num_series} valid series, "
                   f"sum={np.sum(summed):.2f}")
        
        return summed

    summed_real_cumulative_hist = get_summed_cumulative_histogram(
        real_times_per_group, time_bins, "Real")
    summed_pseudo_cumulative_hist = get_summed_cumulative_histogram(
        pseudo_times_per_group, time_bins, "Pseudo")

    # Combine all times for saving/debugging
    combined_real_times_raw = np.concatenate([
        t for t in real_times_per_group if t is not None and len(t) > 0
    ])
    combined_pseudo_times_raw = np.concatenate([
        t for t in pseudo_times_per_group if t is not None and len(t) > 0
    ])

    logger.info(f"    Combined: Real n={len(combined_real_times_raw)}, "
               f"Pseudo n={len(combined_pseudo_times_raw)}")

    return (summed_real_cumulative_hist, summed_pseudo_cumulative_hist, time_bins,
            combined_real_times_raw, combined_pseudo_times_raw, n_groups, N_RANDOM_2)


def determine_characteristic_interaction_time(summed_real_hist, summed_pseudo_hist, 
                                              time_bins, n_groups, n_pseudo, logger):
    """
    Determines the characteristic interaction time from the distributions.
    
    CRITICAL FIX: Proper normalization by n_groups and n_pseudo.
    """
    # Calculate the key metric: (Real/n_groups) - (Pseudo/n_pseudo)
    time_metric = (summed_real_hist / n_groups) - (summed_pseudo_hist / n_pseudo)
    
    # Focus on first half (as in paper)
    first_half_len = round(len(time_metric) / 2)
    if first_half_len == 0:
        logger.warning("    Time metric first half is empty!")
        return None

    relevant_part_metric = time_metric[:first_half_len]
    
    # Check for positive values
    if not np.any(relevant_part_metric > 0):
        logger.warning("    No positive values in time metric! "
                      "Spatial criteria may be too loose.")
        return None

    # Find maximum in first half
    char_time_idx = np.where(relevant_part_metric == np.max(relevant_part_metric))[0][0]
    
    logger.info(f"    Initial char_time: {time_bins[char_time_idx]:.3f}s "
               f"(metric value: {time_metric[char_time_idx]:.6f})")

    # Refinement loop: extend while mean increases
    keep_refining = True
    refinement_steps = 0
    
    while keep_refining:
        if char_time_idx + 1 >= len(time_metric):
            keep_refining = False
            break

        mean_up_to_current = np.mean(time_metric[:char_time_idx + 1])
        mean_up_to_next = np.mean(time_metric[:char_time_idx + 2])

        if mean_up_to_current < mean_up_to_next:
            char_time_idx += 1
            refinement_steps += 1
        else:
            keep_refining = False

        if char_time_idx >= len(time_metric) - 1:
            char_time_idx = len(time_metric) - 1
            keep_refining = False
        
        # Safety check
        if refinement_steps > 500:
            logger.warning("    Refinement exceeded 500 steps!")
            break

    logger.info(f"    After {refinement_steps} refinement steps: {time_bins[char_time_idx]:.3f}s")

    # Final step: find where real cumulative drops to half of value at char_time_idx
    
    if char_time_idx >= len(summed_real_hist):
        char_time_idx = len(summed_real_hist) - 1

    value_at_char_time = summed_real_hist[char_time_idx]
    
    if value_at_char_time <= 0:
        logger.warning(f"    Value at char_time <= 0: {value_at_char_time}")
        return None

    # Find where real cumsum drops below half threshold
    threshold = value_at_char_time * 0.5
    significant_indices = np.where(summed_real_hist < threshold)[0]

    if not significant_indices.size:
        logger.warning("    No crossing point found!")
        return None

    final_time_idx = significant_indices[0]
    characteristic_time = time_bins[final_time_idx]
    
    logger.info(f"    Final characteristic time: {characteristic_time:.3f}s")
    
    # VALIDATION: Warn if value seems unreasonable
    if characteristic_time > 2.0:
        logger.warning(f"    ⚠ Time {characteristic_time:.2f}s > 2.0s! "
                      "Expected range for Canton-S males: 0.4-0.8s. "
                      "Spatial criteria may be too permissive.")
    elif characteristic_time < 0.1:
        logger.warning(f"    ⚠ Time {characteristic_time:.2f}s < 0.1s! "
                      "Value seems too short. Check spatial criteria.")

    return characteristic_time


# ============================================================================
# VALIDATION
# ============================================================================
def validate_criteria(distance, angle, time_val, logger):
    """Validate that criteria are biologically reasonable."""
    issues = []
    
    # Distance validation (typical range: 1-6 body lengths)
    if not (0.5 <= distance <= 10.0):
        issues.append(f"Distance {distance:.2f} outside reasonable range [0.5, 10]")
    
    # Angle validation
    if not (0 <= angle <= 180):
        issues.append(f"Angle {angle:.2f} outside valid range [0, 180]")
    
    # Time validation (typical range: 0.2-5s)
    if not (0.05 <= time_val <= 10.0):
        issues.append(f"Time {time_val:.2f} outside reasonable range [0.05, 10]")
    
    # Warn about values outside typical range
    if distance > 5.0:
        logger.warning(f"  Distance {distance:.2f} > 5.0 BL (high)")
    if angle > 175:
        logger.warning(f"  Angle {angle:.2f}° > 175° (very wide)")
    if time_val > 1.5:
        logger.warning(f"  Time {time_val:.2f}s > 1.5s (long for Canton-S males)")
    
    if issues:
        for issue in issues:
            logger.warning(f"  Validation: {issue}")
    
    return len(issues) == 0


def check_convergence(criteria_df, window=50, logger=None):
    """Check if estimates have converged."""
    if len(criteria_df) < window:
        return False
    
    recent = criteria_df.tail(window)
    
    # Calculate coefficient of variation for each parameter
    cvs = {}
    for param in ['distance', 'angle', 'time']:
        mean_val = recent[param].mean()
        std_val = recent[param].std()
        cvs[param] = std_val / mean_val if mean_val > 0 else float('inf')
    
    threshold = 0.05  # 5% CV indicates convergence
    converged = all(cv < threshold for cv in cvs.values())
    
    if converged and logger:
        logger.info(f"  Convergence detected! CVs: " + 
                   ", ".join(f"{k}={v:.3f}" for k, v in cvs.items()))
    
    return converged


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution loop for estimating interaction criteria."""
    
    # Setup
    logger = setup_logger(settings.LOGS_DIR)
    logger.info("="*70)
    logger.info("DROSOPHILA SOCIAL INTERACTION CRITERIA ESTIMATION")
    logger.info("="*70)
    logger.info(f"Treatment: {settings.TREATMENT}")
    logger.info(f"Random group size: {RANDOM_GROUP_SIZE}")
    logger.info(f"Pseudo iterations: {N_RANDOM_2}")
    logger.info("")
    
    # Load data
    try:
        treatment_data = load_app_data(
            settings.TRACKINGS, 
            settings.NROMALIZATION, 
            settings.PXPERMM,
            logger
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return
    
    # Setup output paths
    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    criteria_output_path = output_dir / f'{settings.TREATMENT}_criteria.csv'
    times_data_base_path = output_dir / settings.TREATMENT / "times"
    times_data_base_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize or load existing criteria
    criteria_df, iteration_count = initialize_or_load_criteria_df(
        criteria_output_path, settings.TREATMENT, logger
    )
    
    # Get coordinate bins
    coord_bins = get_coordinate_bins()
    
    # Tracking
    consecutive_failures = 0
    max_consecutive_failures = 20
    
    # Main iteration loop
    logger.info("")
    logger.info("Starting estimation loop...")
    logger.info("="*70)
    
    while len(criteria_df) < MAX_ITERATIONS:
        loop_iter_start_time = time.time()
        current_iter_label = f"Iteration {iteration_count}"
        
        logger.info(f"\n{current_iter_label} (Valid estimates: {len(criteria_df)}/{MAX_ITERATIONS})")
        
        try:
            # Check if we have enough groups
            if len(treatment_data) < RANDOM_GROUP_SIZE:
                logger.error(
                    f"Not enough treatment groups ({len(treatment_data)}) "
                    f"to sample {RANDOM_GROUP_SIZE}. Stopping."
                )
                break
            
            # Check for convergence every 50 iterations after minimum of 100
            if len(criteria_df) >= 100 and len(criteria_df) % 50 == 0:
                if check_convergence(criteria_df, CONVERGENCE_WINDOW, logger):
                    logger.info("Convergence achieved! Stopping early.")
                    break
            
            # ================================================================
            # STEP 1: SPATIAL ANALYSIS
            # ================================================================
            random_group_indices = random.sample(range(len(treatment_data)), RANDOM_GROUP_SIZE)
            
            logger.info("Step 1: Spatial analysis...")
            super_N_hist, pseudo_N_hist_sample = calculate_base_spatial_metrics(
                treatment_data, random_group_indices, logger
            )
            
            initial_density_map = calculate_initial_social_density_map(
                super_N_hist, pseudo_N_hist_sample, logger
            )

            processed_map = process_spatial_density_map(initial_density_map, coord_bins, logger)
            
            if processed_map is None:
                logger.warning(f"  {current_iter_label}: Spatial map processing failed.")
                consecutive_failures += 1
                iteration_count += 1
                continue

            # ================================================================
            # STEP 2: DETERMINE ANGLE & DISTANCE
            # ================================================================
            logger.info("Step 2: Determining angle and distance...")
            est_angle, est_distance = determine_max_angle_distance_from_map(
                processed_map, coord_bins
            )
            
            if est_angle is None or est_distance is None:
                logger.warning(f"  {current_iter_label}: Could not estimate angle/distance.")
                consecutive_failures += 1
                iteration_count += 1
                continue

            # Refine the estimates
            refinement_density_map = (super_N_hist / RANDOM_GROUP_SIZE) - \
                                    (pseudo_N_hist_sample / N_RANDOM_2)

            final_angle, final_distance = refine_angle_distance_iteratively(
                refinement_density_map, coord_bins, est_angle, est_distance, logger
            )

            if final_angle is None or final_distance is None or final_angle == 0 or final_distance == 0:
                logger.warning(f"  {current_iter_label}: Invalid angle/distance after refinement.")
                consecutive_failures += 1
                iteration_count += 1
                continue

            # ================================================================
            # STEP 3: DETERMINE TIME THRESHOLD
            # ================================================================
            logger.info("Step 3: Determining time threshold...")
            
            try:
                (real_hist, pseudo_hist, time_b, combined_real_t, combined_pseudo_t,
                 n_groups, n_pseudo) = calculate_interaction_time_distributions(
                    treatment_data, random_group_indices, final_angle, final_distance, logger
                )

                char_time = determine_characteristic_interaction_time(
                    real_hist, pseudo_hist, time_b, n_groups, n_pseudo, logger
                )

                # ================================================================
                # STEP 4: VALIDATE AND SAVE
                # ================================================================
                if char_time is not None and char_time > 0:
                    # Validate criteria
                    is_valid = validate_criteria(final_distance, final_angle, char_time, logger)
                    
                    if is_valid:
                        # Save to dataframe
                        new_row = pd.DataFrame([{
                            "distance": final_distance, 
                            "angle": final_angle, 
                            "time": char_time
                        }])
                        criteria_df = pd.concat([criteria_df, new_row], ignore_index=True)
                        
                        # Save periodically
                        if len(criteria_df) % 10 == 0:
                            criteria_df.to_csv(criteria_output_path)
                            logger.info(f"  Saved checkpoint at {len(criteria_df)} estimates")
                        
                        # Save time arrays for debugging
                        np.save(
                            times_data_base_path / f"{iteration_count}_real_array.npy", 
                            combined_real_t
                        )
                        np.save(
                            times_data_base_path / f"{iteration_count}_pseudo_array.npy", 
                            combined_pseudo_t
                        )

                        duration = time.time() - loop_iter_start_time
                        logger.info(
                            f"  ✓ SUCCESS: D={final_distance:.2f} BL, "
                            f"A={final_angle:.1f}°, T={char_time:.2f}s ({duration:.1f}s)"
                        )
                        
                        consecutive_failures = 0
                    else:
                        logger.warning(f"  {current_iter_label}: Criteria validation failed")
                        consecutive_failures += 1
                else:
                    duration = time.time() - loop_iter_start_time
                    logger.warning(
                        f"  {current_iter_label}: Failed to determine characteristic time "
                        f"({duration:.1f}s)"
                    )
                    consecutive_failures += 1

            except Exception as e_time:
                duration = time.time() - loop_iter_start_time
                logger.error(
                    f"  {current_iter_label}: Error in time analysis: {e_time} ({duration:.1f}s)",
                    exc_info=True
                )
                consecutive_failures += 1

            iteration_count += 1
            
            # Check for too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"\nStopping: {consecutive_failures} consecutive failures. "
                    "Check data quality and parameters."
                )
                break

        except Exception as e_outer:
            duration = time.time() - loop_iter_start_time
            logger.error(
                f"  {current_iter_label}: Critical error: {e_outer} ({duration:.1f}s)", 
                exc_info=True
            )
            consecutive_failures += 1
            iteration_count += 1
            
            if consecutive_failures >= max_consecutive_failures:
                break

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("")
    logger.info("="*70)
    logger.info("ESTIMATION COMPLETE")
    logger.info("="*70)
    
    # Final save
    criteria_df.to_csv(criteria_output_path)
    logger.info(f"Results saved to: {criteria_output_path}")
    
    logger.info(f"\nTotal iterations attempted: {iteration_count}")
    logger.info(f"Valid estimates obtained: {len(criteria_df)}")
    logger.info(f"Success rate: {len(criteria_df)/iteration_count*100:.1f}%")
    
    if len(criteria_df) > 0:
        logger.info(f"\n{'='*70}")
        logger.info("FINAL CRITERIA (Median with 95% CI)")
        logger.info(f"{'='*70}")
        
        for param in ['distance', 'angle', 'time']:
            median = criteria_df[param].median()
            ci_lower = criteria_df[param].quantile(0.025)
            ci_upper = criteria_df[param].quantile(0.975)
            
            logger.info(
                f"  {param.capitalize():10s}: {median:6.2f}  "
                f"({ci_lower:6.2f} - {ci_upper:6.2f})"
            )
        
        # Compare to expected values for Canton-S males
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON TO PAPER (Canton-S males)")
        logger.info(f"{'='*70}")
        logger.info("  Distance  :   2.50  ( 2.00 -  3.00) body lengths")
        logger.info("  Angle     : 160.00  (145.00 - 170.00) degrees")
        logger.info("  Time      :   0.60  ( 0.50 -  0.80) seconds")
        
        # Warnings
        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION")
        logger.info(f"{'='*70}")
        
        warnings = []
        if criteria_df['distance'].median() > 4.0:
            warnings.append("⚠ Distance median > 4.0 BL (may be too high)")
        if criteria_df['angle'].median() > 175:
            warnings.append("⚠ Angle median > 175° (very wide)")
        if criteria_df['time'].median() > 1.5:
            warnings.append("⚠ Time median > 1.5s (higher than typical 0.4-1.1s)")
        if criteria_df['time'].median() < 0.3:
            warnings.append("⚠ Time median < 0.3s (lower than typical)")
        
        if warnings:
            for warning in warnings:
                logger.info(f"  {warning}")
        else:
            logger.info("  ✓ All criteria within expected ranges")
    
    logger.info(f"\n{'='*70}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()