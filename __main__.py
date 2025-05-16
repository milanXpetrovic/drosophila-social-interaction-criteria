# %%

import json
import logging
import multiprocessing
import os
import random
import time

import natsort
import numpy as np
import pandas as pd
import src.utils.fileio as fileio
import src.utils.utils as SL
from scipy.signal import convolve2d
from skimage import measure as skimage_label
from src import settings

ANGLE_BIN = settings.ANGLE_BIN
DISTANCE_BIN = settings.DISTANCE_BIN
START_TIME = settings.START
# TIMECUT = settings.TIMECUT # Not used in the provided main logic
EXP_DURATION = settings.EXP_DURATION
RANDOM_GROUP_SIZE = settings.RANDOM_GROUP_SIZE
# N_RANDOM_1 = settings.N_RANDOM_1 # Seems related to 'storeN', which is not critical for criteria_df
N_RANDOM_2 = settings.N_RANDOM_2
DISTANCE_MAX = settings.DISTANCE_MAX
# Derived from original array sizes like angle = np.zeros((500,1))
MAX_ITERATIONS = 500


def setup_logger(log_dir, log_file_name="runtime_log.txt"):
    """Sets up a logger for runtime messages."""
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)

    logger_instance = logging.getLogger("runtime_logger")
    logger_instance.setLevel(logging.INFO)

    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger_instance.addHandler(file_handler)
    return logger_instance


def load_app_data(trackings_path, normalization_path, pxpermm_path):
    """Loads treatment data, normalization, and pixel-per-mm information."""
    treatment_data = fileio.load_multiple_folders(trackings_path)
    sorted_keys = natsort.natsorted(treatment_data.keys())
    treatment_data = {k: treatment_data[k] for k in sorted_keys}

    with open(normalization_path, 'r') as f:
        json.load(f)
    with open(pxpermm_path, 'r') as f:
        json.load(f)

    return treatment_data


def initialize_or_load_criteria_df(save_path, treatment_name, logger_instance):
    """Initializes a new DataFrame for criteria or loads an existing one."""
    if os.path.exists(save_path):
        criteria_df = pd.read_csv(save_path, index_col=0)
        start_index = len(criteria_df)
        logger_instance.info(
            f'{treatment_name} - Continue from: {start_index}')
    else:
        criteria_df = pd.DataFrame(columns=["distance", "angle", "time"])
        start_index = 0
        logger_instance.info(f'{treatment_name} - Start from 0')
    return criteria_df, start_index


def get_coordinate_bins():
    """Generates coordinate bins for distance and angle."""
    distance_coords = np.arange(0, DISTANCE_MAX, DISTANCE_BIN)
    angle_coords = np.arange(-180, 181, ANGLE_BIN)
    return {0: distance_coords, 1: angle_coords}


def calculate_base_spatial_metrics(treatment_data, random_indices):
    """Calculates histograms for all groups and a pseudo-group."""
    treatment_items = list(treatment_data.items())
    with multiprocessing.Pool() as pool:
        all_group_hists = pool.starmap(SL.process_norm_group, treatment_items)

    super_N_hist_total = np.sum(all_group_hists, axis=0)
    pseudo_N_hist_sample = SL.boot_pseudo_fly_space(
        treatment_data, random_indices)

    return super_N_hist_total, pseudo_N_hist_sample


def calculate_initial_social_density_map(super_N_hist, pseudo_N_hist):
    """Calculates the initial N2 density map for identifying regions."""
    if np.sum(super_N_hist) == 0 or np.sum(pseudo_N_hist) == 0:
        return np.zeros_like(super_N_hist)

    density_map = (super_N_hist / np.sum(super_N_hist)) - (pseudo_N_hist / np.sum(pseudo_N_hist))

    falloff = np.arange(1, density_map.shape[0] + 1).astype(float) ** -1
    density_map = density_map * np.tile(falloff, (density_map.shape[1], 1)).T

    positive_density_values = density_map[density_map > 0]
    if len(positive_density_values) > 0:
        threshold = np.percentile(positive_density_values, 95)
        density_map[density_map <= threshold] = 0
    else:
        density_map[:] = 0

    return density_map


def filter_map_by_central_region(target_map, labeled_components, coord_bins):
    """Filters a map, keeping only components overlapping a defined central region."""
    distance_coords = coord_bins[0]
    angle_coords = coord_bins[1]

    bcenter_candidates = np.where(distance_coords < 2)[0]
    if not bcenter_candidates.size:
        return target_map
    bcenter = bcenter_candidates[-5:]

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


def process_spatial_density_map(density_map, coord_bins):
    """Applies convolution, labeling, and filtering to the density map."""
    if not np.any(density_map > 0):
        return None

    h_kernel = np.array([
        [0.0181, 0.0492, 0.0492, 0.0181],
        [0.0492, 0.1336, 0.1336, 0.0492],
        [0.0492, 0.1336, 0.1336, 0.0492],
        [0.0181, 0.0492, 0.0492, 0.0181],
    ])
    h_kernel /= np.sum(h_kernel)
    processed_map = convolve2d(density_map, h_kernel, mode="same")

    binary_map_1 = np.where(processed_map > 0, 1, 0)
    labeled_image_1, num_labels_1 = skimage_label.label(
        binary_map_1, connectivity=2, return_num=True)
    if num_labels_1 == 0:
        return None

    cc_1 = {"NumObjects": num_labels_1,
            "PixelIdxList": [np.where(labeled_image_1 == i) for i in range(1, num_labels_1 + 1)]}

    map_after_central_filter_1 = filter_map_by_central_region(processed_map, cc_1, coord_bins)
    if not np.any(map_after_central_filter_1 > 0):
        return None

    positive_values = map_after_central_filter_1[map_after_central_filter_1 > 0]
    if not positive_values.size:
        return None

    map_after_central_filter_1[map_after_central_filter_1 < np.percentile(positive_values, 75)] = 0
    if not np.any(map_after_central_filter_1 > 0):
        return None

    binary_map_2 = np.where(map_after_central_filter_1 > 0, 1, 0)
    labeled_image_2, num_labels_2 = skimage_label.label(binary_map_2, connectivity=2, return_num=True)
    if num_labels_2 == 0:
        return None

    cc_2 = {"NumObjects": num_labels_2, "PixelIdxList": [
        np.where(labeled_image_2 == i) for i in range(1, num_labels_2 + 1)]}

    n2_snapshot = np.copy(map_after_central_filter_1)
    final_map_attempt = filter_map_by_central_region(n2_snapshot, cc_2, coord_bins)

    if not np.any(final_map_attempt > 0):
        final_map_attempt = np.copy(n2_snapshot)
        for i in range(cc_2["NumObjects"]):
            if len(cc_2["PixelIdxList"][i][0]) < 5:
                final_map_attempt[cc_2["PixelIdxList"][i]] = 0

    if not np.any(final_map_attempt > 0):
        return None

    return final_map_attempt


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


def refine_angle_distance_iteratively(density_map_for_refinement, coord_bins, initial_angle, initial_distance):
    """Iteratively refines angle and distance based on mean densities in expanding regions."""
    mean_overall_density = np.mean(density_map_for_refinement)
    distance_coords = coord_bins[0]
    angle_coords = coord_bins[1]

    refined_angle = float(initial_angle)
    refined_distance = float(initial_distance)

    keep_iterating = True
    while keep_iterating:
        try:
            dist_start_idx = np.where(distance_coords == 1)[0]
            if not dist_start_idx.size:
                break
            dist_start_idx = dist_start_idx[0]

            dist_end_idx = np.where(distance_coords == refined_distance)[0][0] + 1
            ang_start_idx = np.where(angle_coords == -refined_angle)[0][0]
            ang_end_idx = np.where(angle_coords == refined_angle)[0][0] + 1

            current_region_slice = density_map_for_refinement[dist_start_idx:dist_end_idx, ang_start_idx:ang_end_idx]
            if current_region_slice.size == 0:
                break
            mean_current_region = current_region_slice.mean()

            ang_expanded_coords_idx = np.where((angle_coords >= -refined_angle - ANGLE_BIN)
                                               & (angle_coords <= refined_angle + ANGLE_BIN))[0]
            dist_expanded_coord_idx = np.where(distance_coords == refined_distance + DISTANCE_BIN)[0]

            if not (ang_expanded_coords_idx.size and dist_expanded_coord_idx.size):
                break

            ang_exp_start_idx, ang_exp_end_idx = ang_expanded_coords_idx[0], ang_expanded_coords_idx[-1] + 1
            dist_exp_end_idx = dist_expanded_coord_idx[0] + 1

            angle_expanded_slice = density_map_for_refinement[dist_start_idx:dist_end_idx,
                                                              ang_exp_start_idx:ang_exp_end_idx]
            distance_expanded_slice = density_map_for_refinement[
                dist_start_idx:dist_exp_end_idx, ang_start_idx:ang_end_idx]
            both_expanded_slice = density_map_for_refinement[dist_start_idx:
                                                             dist_exp_end_idx, ang_exp_start_idx:ang_exp_end_idx]

            if angle_expanded_slice.size == 0 or distance_expanded_slice.size == 0 or both_expanded_slice.size == 0:
                break

            mean_angle_expanded = angle_expanded_slice.mean()
            mean_distance_expanded = distance_expanded_slice.mean()
            mean_both_expanded = both_expanded_slice.mean()

            update_type = 0
            if mean_both_expanded > mean_angle_expanded and mean_both_expanded > mean_distance_expanded:
                if (both_expanded_slice.size * mean_overall_density) > np.sum(angle_expanded_slice):
                    update_type = 3
            elif mean_angle_expanded > mean_distance_expanded:
                if (angle_expanded_slice.size * mean_overall_density > np.sum(angle_expanded_slice)) and \
                   (mean_angle_expanded > mean_current_region):
                    update_type = 1
            else:
                if (distance_expanded_slice.size * mean_overall_density < np.sum(distance_expanded_slice)) and \
                   (mean_distance_expanded > mean_current_region):
                    update_type = 2

            if update_type == 3:
                refined_angle += ANGLE_BIN
                refined_distance += DISTANCE_BIN
            elif update_type == 1:
                refined_angle += ANGLE_BIN
            elif update_type == 2:
                refined_distance += DISTANCE_BIN
            else:
                keep_iterating = False

            if refined_angle > 180:
                refined_angle = 180.0
                keep_iterating = False
            if refined_distance >= DISTANCE_MAX:
                refined_distance = float(DISTANCE_MAX - DISTANCE_BIN)
                keep_iterating = False
            if refined_distance < 0:
                refined_distance = 0.0
                keep_iterating = False

        except (IndexError, ValueError):
            keep_iterating = False

    return refined_angle, refined_distance


def calculate_interaction_time_distributions(treatment_data, random_indices, selected_angle, selected_distance):
    """Calculates real and pseudo cumulative time distributions for interactions."""
    picked_groups_values = [list(treatment_data.values())[i] for i in random_indices]
    pool_args_real = [(group_data, selected_angle, selected_distance) for group_data in picked_groups_values]

    with multiprocessing.Pool() as pool:
        real_times_per_group = list(pool.map(SL.process_group, pool_args_real))

    pseudo_times_per_group = SL.boot_pseudo_times(
        treatment_data, N_RANDOM_2, random_indices, selected_angle, selected_distance, START_TIME, EXP_DURATION)

    time_bins = np.arange(0, 30 * 60 + 0.05, 0.05)

    def get_summed_cumulative_histogram(time_series_list, bins):
        num_series = len(time_series_list)
        hist_matrix = np.zeros((num_series, len(bins) - 1))
        for i, series_data in enumerate(time_series_list):
            if series_data is None or len(series_data) <= 1:
                continue

            hist_counts = np.histogram(series_data[:-1], bins=bins)[0]
            if np.sum(hist_counts) == 0:
                continue

            normalized_hist = hist_counts / np.sum(hist_counts)
            cumulative_sum_rev = np.cumsum(normalized_hist[::-1])[::-1]
            hist_matrix[i, :] = cumulative_sum_rev

        hist_matrix[np.isnan(hist_matrix)] = 0
        return np.sum(hist_matrix, axis=0)

    summed_real_cumulative_hist = get_summed_cumulative_histogram(real_times_per_group, time_bins)
    summed_pseudo_cumulative_hist = get_summed_cumulative_histogram(pseudo_times_per_group, time_bins)

    combined_real_times_raw = np.concatenate([t for t in real_times_per_group if t is not None and len(t) > 0])
    combined_pseudo_times_raw = np.concatenate([t for t in pseudo_times_per_group if t is not None and len(t) > 0])

    return summed_real_cumulative_hist, summed_pseudo_cumulative_hist, time_bins, \
        combined_real_times_raw, combined_pseudo_times_raw


def determine_characteristic_interaction_time(summed_real_hist, summed_pseudo_hist, time_bins):
    """Determines the characteristic interaction time from the distributions."""
    time_metric = (summed_real_hist / RANDOM_GROUP_SIZE) - (summed_pseudo_hist / N_RANDOM_2)

    first_half_len = round(len(time_metric) / 2)
    if first_half_len == 0:
        return None

    relevant_part_metric = time_metric[:first_half_len]
    if not relevant_part_metric.size or np.all(relevant_part_metric <= 0):
        return None

    char_time_idx = np.where(relevant_part_metric == np.max(relevant_part_metric))[0][0]

    keep_refining = True
    while keep_refining:
        if char_time_idx + 1 >= len(time_metric):
            keep_refining = False
            break

        mean_up_to_current = np.mean(time_metric[:char_time_idx + 1])
        mean_up_to_next = np.mean(time_metric[:char_time_idx + 2])

        if mean_up_to_current < mean_up_to_next:
            char_time_idx += 1
        else:
            keep_refining = False

        if char_time_idx >= len(time_metric) - 1:
            char_time_idx = len(time_metric) - 1
            keep_refining = False

    if char_time_idx >= len(summed_real_hist):
        char_time_idx = len(summed_real_hist) - 1

    value_at_char_time = summed_real_hist[char_time_idx]
    if value_at_char_time <= 0:
        return None

    significant_indices = np.where(
        summed_real_hist * 0.5 < value_at_char_time)[0]

    if not significant_indices.size:
        return None

    final_time_idx = significant_indices[0]
    characteristic_time = time_bins[final_time_idx]
    return characteristic_time


def main():
    logger = setup_logger(settings.LOGS_DIR)

    treatment_data = load_app_data(settings.TRACKINGS, settings.NROMALIZATION, settings.PXPERMM)
    criteria_output_path = os.path.join(settings.OUTPUT_DIR, f'{settings.TREATMENT}_criteria.csv')
    criteria_df, iteration_count = initialize_or_load_criteria_df(criteria_output_path, settings.TREATMENT, logger)

    coord_bins = get_coordinate_bins()

    while len(criteria_df) < MAX_ITERATIONS:
        loop_iter_start_time = time.time()
        current_iter_label = f"Iter {iteration_count} (Target: {len(criteria_df)})"
        try:
            if len(treatment_data) < RANDOM_GROUP_SIZE:
                logger.error(
                    f"Not enough treatment groups ({len(treatment_data)}) to sample {RANDOM_GROUP_SIZE}. Stopping.")
                break

            random_group_indices = random.sample(range(len(treatment_data)), RANDOM_GROUP_SIZE)
            super_N_hist, pseudo_N_hist_sample = calculate_base_spatial_metrics(treatment_data, random_group_indices)
            initial_density_map = calculate_initial_social_density_map(super_N_hist, pseudo_N_hist_sample)

            processed_map = process_spatial_density_map(initial_density_map, coord_bins)
            if processed_map is None:
                logger.warning(
                    f"{settings.TREATMENT} - {current_iter_label}: Spatial map processing yielded no significant regions.")
                iteration_count += 1
                continue

            est_angle, est_distance = determine_max_angle_distance_from_map(processed_map, coord_bins)
            if est_angle is None or est_distance is None:
                logger.warning(
                    f"{settings.TREATMENT} - {current_iter_label}: Could not estimate initial angle/distance from processed map.")
                iteration_count += 1
                continue

            refinement_density_map = (super_N_hist / RANDOM_GROUP_SIZE) - (pseudo_N_hist_sample / N_RANDOM_2)

            final_angle, final_distance = refine_angle_distance_iteratively(
                refinement_density_map, coord_bins, est_angle, est_distance)

            if final_angle is None or final_distance is None or final_angle == 0 or final_distance == 0:
                logger.warning(
                    f"{settings.TREATMENT} - {current_iter_label}: Angle/Distance refinement resulted in zero/invalid values.")
                iteration_count += 1
                continue

            try:
                real_hist, pseudo_hist, time_b, combined_real_t, combined_pseudo_t = \
                    calculate_interaction_time_distributions(
                        treatment_data, random_group_indices, final_angle, final_distance)

                char_time = determine_characteristic_interaction_time(real_hist, pseudo_hist, time_b)

                if char_time is not None and char_time > 0:
                    new_row = pd.DataFrame([{"distance": final_distance, "angle": final_angle, "time": char_time}])
                    criteria_df = pd.concat([criteria_df, new_row], ignore_index=True)
                    criteria_df.to_csv(criteria_output_path)

                    times_data_base_path = f"/srv/milky/drosophila-datasets/{settings.TREATMENT}"
                    os.makedirs(times_data_base_path, exist_ok=True)
                    np.save(os.path.join(times_data_base_path, f"{iteration_count}_real_array.npy"), combined_real_t)
                    np.save(os.path.join(times_data_base_path,
                            f"{iteration_count}_pseudo_array.npy"), combined_pseudo_t)

                    duration = time.time() - loop_iter_start_time
                    logger.info(
                        f"{settings.TREATMENT} - {current_iter_label}: D={final_distance:.2f}, A={final_angle:.2f}, T={char_time:.2f}. Took {duration:.2f}s")
                else:
                    duration = time.time() - loop_iter_start_time
                    logger.warning(
                        f"{settings.TREATMENT} - {current_iter_label}: Failed to determine characteristic time. Took {duration:.2f}s")

            except Exception as e_time:
                duration = time.time() - loop_iter_start_time
                logger.error(
                    f"{settings.TREATMENT} - {current_iter_label}: Error in time analysis: {e_time}. Took {duration:.2f}s")

            iteration_count += 1

        except Exception as e_outer:
            duration = time.time() - loop_iter_start_time
            logger.error(
                f"{settings.TREATMENT} - {current_iter_label}: Critical error in iteration: {e_outer}. Took {duration:.2f}s")
            iteration_count += 1

    logger.info(
        f"Processing finished. Total criteria rows generated: {len(criteria_df)} after {iteration_count} attempts.")


if __name__ == '__main__':
    main()
