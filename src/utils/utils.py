import itertools
import json
import multiprocessing
import os
import random

import natsort
import numpy as np
import pandas as pd
import src.utils.fileio as fileio  # Assuming this exists and is correct
from scipy.signal import \
    find_peaks  # Retained, though not used by main.py functions directly
from src import settings  # Assuming this provides necessary constants

# --- Cached Global Data ---
_NORMALIZATION_DATA_CACHE = None
_PXPERMM_DATA_CACHE = None


def _load_global_json_data():
    """Loads normalization and pxpermm data into global caches if not already loaded."""
    global _NORMALIZATION_DATA_CACHE, _PXPERMM_DATA_CACHE
    if _NORMALIZATION_DATA_CACHE is None:
        with open(settings.NROMALIZATION, 'r') as f:
            _NORMALIZATION_DATA_CACHE = json.load(f)
    if _PXPERMM_DATA_CACHE is None:
        with open(settings.PXPERMM, 'r') as f:
            _PXPERMM_DATA_CACHE = json.load(f)

# Ensure data is loaded when module is imported or first used by a process
# _load_global_json_data() # Call it here, or ensure first function needing it calls it.

# --- Core Utility Functions ---


def angledifference_nd(angle1_deg, angle2_deg):
    diff = angle2_deg - angle1_deg
    diff = (diff + 180) % 360 - 180
    return diff


def rotation(input_xy, center_xy, anti_clockwise_angle_deg):
    if input_xy.shape[1] != 2:
        raise ValueError("Input coordinates must have 2 columns (x, y).")
    if not (isinstance(center_xy, (list, tuple, np.ndarray)) and len(center_xy) == 2):
        raise ValueError("Center must be a 2-element list, tuple, or array (x, y).")

    center_arr = np.array(center_xy).reshape(1, 2)
    angle_rad = np.deg2rad(anti_clockwise_angle_deg)

    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotation_matrix = np.array([[cos_angle, -sin_angle],
                                [sin_angle,  cos_angle]])

    translated_points = input_xy - center_arr
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    return rotated_points + center_arr

# --- Data Normalization and Preparation ---


def _load_and_normalize_fly_data_from_path(fly_path, norm_params, pxpermm_val_for_group):
    npy_data = np.load(fly_path)
    # Assuming npy_data[:, 0] are frame indices, data starts from npy_data[:, 1]
    df = pd.DataFrame(npy_data[:, 1:], columns=["pos x", "pos y", "ori", "a", "b"], index=npy_data[:, 0])

    df["pos x"] = (df["pos x"] - norm_params["x"] + norm_params["radius"]) / (2 * norm_params["radius"])
    df["pos y"] = (df["pos y"] - norm_params["y"] + norm_params["radius"]) / (2 * norm_params["radius"])
    df["a"] = df["a"] / (2 * norm_params["radius"])

    normalized_pxpermm = pxpermm_val_for_group / (2 * norm_params["radius"])
    return df, normalized_pxpermm


def normalize_group(group_data_paths, is_pseudo):
    _load_global_json_data()  # Ensure JSONs are loaded
    normalization_all_groups = _NORMALIZATION_DATA_CACHE
    pxpermm_all_groups = _PXPERMM_DATA_CACHE

    normalized_dfs_map = {}
    pxpermm_map = {}

    if is_pseudo:
        num_groups_to_sample = 12
        available_group_keys = list(group_data_paths.keys())
        num_to_sample = min(num_groups_to_sample, len(available_group_keys))

        sampled_group_keys = random.sample(available_group_keys, num_to_sample)

        for group_key_orig in sampled_group_keys:
            group_folder_path = group_data_paths[group_key_orig]
            norm_params = normalization_all_groups[group_key_orig]
            pxpermm_val_for_group = pxpermm_all_groups[group_key_orig]

            num_flies_in_group = 12  # Consider making this dynamic if possible
            fly_to_sample_idx = random.randint(1, num_flies_in_group)
            fly_file_name = f"fly{fly_to_sample_idx}.npy"
            fly_path = os.path.join(group_folder_path, fly_file_name)

            if os.path.exists(fly_path):
                df, norm_pxpermm = _load_and_normalize_fly_data_from_path(fly_path, norm_params, pxpermm_val_for_group)
                normalized_dfs_map[group_key_orig] = df
                pxpermm_map[group_key_orig] = norm_pxpermm
    else:
        for group_name, group_folder_path in group_data_paths.items():
            norm_params = normalization_all_groups[group_name]
            pxpermm_val_for_group = pxpermm_all_groups[group_name]

            fly_files = fileio.load_files_from_folder(group_folder_path, file_format=".npy")
            for fly_name_in_group, fly_path in fly_files.items():
                df, norm_pxpermm = _load_and_normalize_fly_data_from_path(fly_path, norm_params, pxpermm_val_for_group)
                normalized_dfs_map[fly_name_in_group] = df
                pxpermm_map[fly_name_in_group] = norm_pxpermm

    return normalized_dfs_map, pxpermm_map


def get_trx(normalized_dfs, pxpermm_map, apply_random_rotation=True):
    trx = {}
    for fly_key, fly_df in normalized_dfs.items():
        x = fly_df["pos x"].to_numpy()
        y = fly_df["pos y"].to_numpy()
        coords = np.column_stack((x, y))

        if apply_random_rotation:
            random_angle_deg = np.random.uniform(0, 360)
            coords = rotation(coords, [0.5, 0.5], random_angle_deg)

        dict_values = {
            "pos x": coords[:, 0],
            "pos y": coords[:, 1],
            "ori": fly_df["ori"].to_numpy(),  # Should be in radians
            "a": fly_df["a"].to_numpy(),
            "pxpermm": pxpermm_map[fly_key],
        }
        # Store as DataFrame, consistent with original return, though NumPy arrays might be used later
        trx[fly_key] = pd.DataFrame(dict_values, index=fly_df.index)
    return trx

# --- Spatial Histogram Calculation ---


def group_space_angle_hist(normalized_dfs, pxpermm_map, is_pseudo):
    hist_angle_edges = np.arange(-177.5, 177.6, settings.ANGLE_BIN)
    hist_dist_edges = np.arange(0.125, settings.DISTANCE_MAX + settings.DISTANCE_BIN/2, settings.DISTANCE_BIN)

    num_angle_hist_bins = len(hist_angle_edges) - 1
    num_dist_hist_bins = len(hist_dist_edges) - 1

    # Shape: (num_angle_bins_from_hist2d + 2 for padding, num_dist_bins)
    accumulated_hist_T_padded = np.zeros((num_angle_hist_bins + 2, num_dist_hist_bins))

    for fly1_key, fly2_key in itertools.permutations(normalized_dfs.keys(), 2):
        df1 = normalized_dfs[fly1_key]
        df2 = normalized_dfs[fly2_key]

        common_index = df1.index.intersection(df2.index)
        if common_index.empty:
            continue

        # Align and convert to numpy once
        # Columns: 0:pos x, 1:pos y, 2:ori(radians), 3:a (normalized)
        df1_aligned = df1.loc[common_index, ["pos x", "pos y", "ori", "a"]].to_numpy()
        df2_aligned = df2.loc[common_index, ["pos x", "pos y"]].to_numpy()  # ori, a not needed for fly2

        fly1_x, fly1_y, fly1_ori_rad, fly1_a_norm = df1_aligned[:,
                                                                0], df1_aligned[:, 1], df1_aligned[:, 2], df1_aligned[:, 3]
        fly2_x, fly2_y = df2_aligned[:, 0], df2_aligned[:, 1]

        mean_body_length_fly1_norm = np.nanmean(fly1_a_norm)
        if mean_body_length_fly1_norm == 0 or np.isnan(mean_body_length_fly1_norm):
            continue

        dx = fly2_x - fly1_x
        dy = fly2_y - fly1_y
        distance_pixels_norm = np.sqrt(dx**2 + dy**2)
        distance_bodylengths = distance_pixels_norm / (mean_body_length_fly1_norm * 4.0)

        angle_to_fly2_deg = np.arctan2(dy, dx) * 180 / np.pi
        relative_angle_deg = angledifference_nd(angle_to_fly2_deg, fly1_ori_rad * 180 / np.pi)

        mask = distance_bodylengths <= settings.DISTANCE_MAX
        # MOVECUT logic from original was complex and seems disabled in main.py context, omitted for clarity.

        hist, _, _ = np.histogram2d(
            relative_angle_deg[mask],
            distance_bodylengths[mask],
            bins=(hist_angle_edges, hist_dist_edges)
        )  # hist shape: (num_angle_hist_bins, num_dist_hist_bins)

        if hist.shape[0] > 0:  # If there are angle bins from histogram2d
            accumulated_hist_T_padded[1:-1, :] += hist
            if hist.shape[0] > 1:
                mean_boundary_row = np.mean(hist[[0, -1], :], axis=0)
            else:
                mean_boundary_row = hist[0, :]  # Only one angle bin
            accumulated_hist_T_padded[0, :] += mean_boundary_row
            accumulated_hist_T_padded[-1, :] += mean_boundary_row

    final_hist = accumulated_hist_T_padded
    max_val = np.max(final_hist)  # Max taken before potential ceiling operation

    if is_pseudo:
        processed_hist = np.ceil(final_hist)
    else:
        if max_val > 0:
            processed_hist = np.ceil((final_hist / max_val) * 256.0)
        else:
            processed_hist = np.ceil(final_hist)

    return processed_hist.T  # Transpose to (dist_bins, angle_bins_padded)


def _process_pseudo_iteration_for_space(args_tuple):
    group_sample_for_iteration, _ = args_tuple
    normalized_dfs, pxpermm_dict = normalize_group(group_sample_for_iteration, is_pseudo=True)
    hist = group_space_angle_hist(normalized_dfs, pxpermm_dict, is_pseudo=True)
    return hist


def boot_pseudo_fly_space(treatment_data, random_group_indices_for_treatment):
    keys = list(treatment_data.keys())
    values = list(treatment_data.values())

    subset_treatment_data = {
        keys[i]: values[i] for i in random_group_indices_for_treatment
    }

    num_pseudo_iterations = 15  # As per original's second version
    pool_args = [(subset_treatment_data, i) for i in range(num_pseudo_iterations)]

    with multiprocessing.Pool() as pool:
        all_pseudo_hists = pool.map(_process_pseudo_iteration_for_space, pool_args)

    return np.sum(all_pseudo_hists, axis=0) if all_pseudo_hists else np.array([])


def process_norm_group(group_name, group_path):
    group_dict = {group_name: group_path}
    normalized_dfs, pxpermm_dict = normalize_group(group_dict, is_pseudo=False)
    hist = group_space_angle_hist(normalized_dfs, pxpermm_dict, is_pseudo=False)
    return hist

# --- Interaction Time Calculation ---


def _calculate_interaction_durations(interaction_frames_indices, fps):
    if interaction_frames_indices.size == 0:
        return np.array([])

    # Find groups of consecutive frames
    diffs = np.diff(interaction_frames_indices)
    bout_ends = np.where(diffs > 1)[0]  # Indices where a jump occurs

    durations_in_frames = []
    start_idx = 0
    for end_idx_in_diffs_array in bout_ends:
        # The bout ends at interaction_frames_indices[end_idx_in_diffs_array]
        # Its length is end_idx_in_diffs_array - start_idx + 1
        durations_in_frames.append(end_idx_in_diffs_array - start_idx + 1)
        start_idx = end_idx_in_diffs_array + 1  # Next bout starts after the jump

    # Add the last bout
    durations_in_frames.append(len(interaction_frames_indices) - start_idx)

    valid_durations = np.array(durations_in_frames)
    return valid_durations[valid_durations > 0] / fps


def _load_and_prepare_fly_trajectory_from_path(fly_path, num_frames_needed, frame_offset_in_file):
    """Loads, slices, and pads trajectory data from an NPY file path."""
    npy_data = np.load(fly_path)  # Assumed: [frame_idx_col, x, y, ori_rad, a, b]

    available_frames_in_file = npy_data.shape[0]
    slice_start = min(frame_offset_in_file, available_frames_in_file)
    slice_end = min(frame_offset_in_file + num_frames_needed, available_frames_in_file)

    window_data = npy_data[slice_start:slice_end, :]

    # Columns for trajectory: x, y, ori (radians)
    raw_trajectory = window_data[:, 1:4] if window_data.size > 0 else np.empty((0, 3))
    # Column for body length 'a'
    raw_a_axis = window_data[:, 4] if window_data.size > 0 else np.empty((0,))

    current_frames_in_window = raw_trajectory.shape[0]

    if current_frames_in_window >= num_frames_needed:
        fly_data = raw_trajectory[:num_frames_needed, :]
        fly_a_values = raw_a_axis[:num_frames_needed]
    else:
        pad_width = num_frames_needed - current_frames_in_window
        fly_data = np.pad(raw_trajectory, ((0, pad_width), (0, 0)),
                          mode='constant', constant_values=np.nan)
        fly_a_values = np.pad(raw_a_axis, (0, pad_width),
                              mode='constant', constant_values=np.nan)

    mean_a = np.nanmean(fly_a_values)
    return fly_data, (0.0 if np.isnan(mean_a) else mean_a)


def fast_flag_interactions(trx_fly_paths, timecut_sec, min_angle_deg, min_dist_bl,
                           start_time_min, exp_duration_min, num_flies, fps, movecut_threshold):

    frame_offset = int(start_time_min * 60 * fps)
    num_frames_to_analyze = int(exp_duration_min * 60 * fps)

    if num_flies == 0 or num_frames_to_analyze == 0:
        return np.array([])

    fly_prepared_data_list = []  # List of dicts: {"trajectory": array, "mean_a": float}
    sorted_fly_ids = natsort.natsorted(list(trx_fly_paths.keys()))

    for fly_id in sorted_fly_ids:
        fly_path = trx_fly_paths[fly_id]
        trajectory_arr, mean_a_val = _load_and_prepare_fly_trajectory_from_path(
            fly_path, num_frames_to_analyze, frame_offset
        )
        fly_prepared_data_list.append({"trajectory": trajectory_arr, "mean_a": mean_a_val})

    distances_matrix = np.zeros((num_flies, num_flies, num_frames_to_analyze))
    angles_matrix = np.zeros((num_flies, num_flies, num_frames_to_analyze))

    distance_thresholds_per_focal = np.array(
        [fpd["mean_a"] * 4.0 * min_dist_bl for fpd in fly_prepared_data_list]
    )
    # Handle cases where mean_a might be 0 or NaN (though _load_and_prepare... returns 0.0 for NaN mean_a)
    distance_thresholds_per_focal[np.isnan(distance_thresholds_per_focal)] = 0.0

    for i in range(num_flies):
        fly1_traj = fly_prepared_data_list[i]["trajectory"]  # Shape (num_frames, 3: x,y,ori_rad)
        for j in range(num_flies):
            if i == j:
                continue
            fly2_traj = fly_prepared_data_list[j]["trajectory"]

            dx = fly2_traj[:, 0] - fly1_traj[:, 0]
            dy = fly2_traj[:, 1] - fly1_traj[:, 1]

            distances_matrix[i, j, :] = np.sqrt(dx**2 + dy**2)

            angle_to_fly2_rad = np.arctan2(dy, dx)
            angles_matrix[i, j, :] = angledifference_nd(
                angle_to_fly2_rad * 180 / np.pi,
                fly1_traj[:, 2] * 180 / np.pi  # fly1_traj ori is in radians
            )

    dist_check = distances_matrix < distance_thresholds_per_focal[:, np.newaxis, np.newaxis]
    angle_check = np.abs(angles_matrix) < min_angle_deg
    interaction_flags = np.logical_and(dist_check, angle_check).astype(np.int8)

    for i in range(num_flies):
        interaction_flags[i, i, :] = 0  # No self-interactions

    all_durations_sec = []
    for i in range(num_flies):
        for j in range(num_flies):
            if i == j:
                continue
            interaction_frames_indices = np.where(interaction_flags[i, j, :] == 1)[0]
            durations_for_pair_sec = _calculate_interaction_durations(interaction_frames_indices, fps)
            if durations_for_pair_sec.size > 0:
                all_durations_sec.extend(durations_for_pair_sec.tolist())

    final_durations = np.array(all_durations_sec)
    return final_durations[final_durations > 0] if final_durations.size > 0 else np.array([])


def process_group(args):
    group_path, selected_angle_deg, selected_distance_bl = args
    trx_fly_paths = fileio.load_files_from_folder(group_path, file_format=".npy")
    num_flies = len(trx_fly_paths)

    if num_flies == 0:
        return np.array([])

    return fast_flag_interactions(
        trx_fly_paths, settings.TIMECUT, selected_angle_deg, selected_distance_bl,
        settings.START, settings.EXP_DURATION, num_flies, settings.FPS, movecut_threshold=0
    )

# --- Pseudo-group Interaction Time Calculation (Optimized for Multiprocessing) ---


def _normalize_pseudo_group_for_time_calc(iteration_index, treatment_subset_paths,
                                          rand_rot_flag, fps,
                                          start_time_min, exp_duration_min):
    """ Normalizes, applies random rotation, slices/pads trajectories for a pseudo-group. """
    normalized_dfs, pxpermm_map = normalize_group(treatment_subset_paths, is_pseudo=True)
    if not normalized_dfs:  # No flies sampled
        return (iteration_index, [], np.array([]), 0, 0)

    trx_dataframes = get_trx(normalized_dfs, pxpermm_map, apply_random_rotation=rand_rot_flag)

    frame_offset = int(start_time_min * 60 * fps)
    num_frames_needed = int(exp_duration_min * 60 * fps)

    pseudo_fly_trajectory_arrays = []
    pseudo_fly_mean_a_values = []

    sorted_pseudo_fly_ids = natsort.natsorted(list(trx_dataframes.keys()))

    for fly_id in sorted_pseudo_fly_ids:
        df = trx_dataframes[fly_id]  # Contains columns 'pos x', 'pos y', 'ori', 'a'

        # Slice DataFrame for the time window (assuming df index is 0-based for its content)
        df_window = df.iloc[frame_offset: frame_offset + num_frames_needed]

        raw_trajectory = df_window[["pos x", "pos y", "ori"]].to_numpy()
        raw_a_axis = df_window["a"].to_numpy()

        current_frames_in_window = raw_trajectory.shape[0]
        if current_frames_in_window >= num_frames_needed:
            trajectory_arr = raw_trajectory[:num_frames_needed, :]
            a_values_arr = raw_a_axis[:num_frames_needed]
        else:
            pad_width = num_frames_needed - current_frames_in_window
            trajectory_arr = np.pad(raw_trajectory, ((0, pad_width), (0, 0)),
                                    mode='constant', constant_values=np.nan)
            a_values_arr = np.pad(raw_a_axis, (0, pad_width),
                                  mode='constant', constant_values=np.nan)

        pseudo_fly_trajectory_arrays.append(trajectory_arr)
        mean_a = np.nanmean(a_values_arr)
        pseudo_fly_mean_a_values.append(0.0 if np.isnan(mean_a) else mean_a)

    num_pseudo_flies = len(pseudo_fly_trajectory_arrays)

    return (iteration_index, pseudo_fly_trajectory_arrays,
            np.array(pseudo_fly_mean_a_values), num_pseudo_flies, num_frames_needed)


def pseudo_fast_flag_interactions(args_from_norm_iter,
                                  min_angle_deg, min_dist_bl, fps):  # Removed start/exp duration, use num_frames_needed
    """ Calculates interaction durations for pre-processed pseudo-group data. """
    _iter_idx, fly_trajectory_arrays, fly_mean_a_np_array, num_flies, num_frames_analyzed = args_from_norm_iter

    if num_flies == 0 or num_frames_analyzed == 0:
        return np.array([])

    distance_thresholds_per_focal = fly_mean_a_np_array * 4.0 * min_dist_bl
    distance_thresholds_per_focal[np.isnan(distance_thresholds_per_focal)] = 0.0

    distances_matrix = np.zeros((num_flies, num_flies, num_frames_analyzed))
    angles_matrix = np.zeros((num_flies, num_flies, num_frames_analyzed))

    for i in range(num_flies):
        fly1_traj = fly_trajectory_arrays[i]
        for j in range(num_flies):
            if i == j:
                continue
            fly2_traj = fly_trajectory_arrays[j]

            dx = fly2_traj[:, 0] - fly1_traj[:, 0]
            dy = fly2_traj[:, 1] - fly1_traj[:, 1]

            distances_matrix[i, j, :] = np.sqrt(dx**2 + dy**2)
            angle_to_fly2_rad = np.arctan2(dy, dx)
            angles_matrix[i, j, :] = angledifference_nd(
                angle_to_fly2_rad * 180 / np.pi,
                fly1_traj[:, 2] * 180 / np.pi  # ori is in radians
            )

    dist_check = distances_matrix < distance_thresholds_per_focal[:, np.newaxis, np.newaxis]
    angle_check = np.abs(angles_matrix) < min_angle_deg
    interaction_flags = np.logical_and(dist_check, angle_check).astype(np.int8)

    for i in range(num_flies):
        interaction_flags[i, i, :] = 0

    all_durations_sec = []
    for i in range(num_flies):
        for j in range(num_flies):
            if i == j:
                continue
            interaction_frames_indices = np.where(interaction_flags[i, j, :] == 1)[0]
            durations_for_pair_sec = _calculate_interaction_durations(interaction_frames_indices, fps)
            if durations_for_pair_sec.size > 0:
                all_durations_sec.extend(durations_for_pair_sec.tolist())

    final_durations = np.array(all_durations_sec)
    return final_durations[final_durations > 0] if final_durations.size > 0 else np.array([])


def boot_pseudo_times(treatment_data_all, num_random_iterations_nrand2,
                      random_group_indices_for_pool,
                      selected_angle_deg, selected_distance_bl,
                      start_time_min, exp_duration_min):

    keys = list(treatment_data_all.keys())
    values = list(treatment_data_all.values())
    treatment_subset_paths = {
        keys[i]: values[i] for i in random_group_indices_for_pool
    }

    apply_random_rotation_to_pseudo = True
    fps = settings.FPS

    norm_pool_args = [
        (pi, treatment_subset_paths, apply_random_rotation_to_pseudo, fps,
         start_time_min, exp_duration_min)
        for pi in range(num_random_iterations_nrand2)
    ]

    with multiprocessing.Pool() as pool:
        # Stage 1: Normalize, rotate, slice/pad trajectories
        # Returns list of (iter_idx, trajectory_arrays_list, mean_a_array, nflies, num_frames_analyzed)
        prepared_pseudo_groups_data = pool.starmap(_normalize_pseudo_group_for_time_calc, norm_pool_args)

        # Stage 2: Calculate interaction times
        interaction_pool_args = [
            (prep_data_tuple, selected_angle_deg, selected_distance_bl, fps)
            for prep_data_tuple in prepared_pseudo_groups_data
        ]
        list_of_time_arrays = pool.starmap(pseudo_fast_flag_interactions, interaction_pool_args)

    return list_of_time_arrays
