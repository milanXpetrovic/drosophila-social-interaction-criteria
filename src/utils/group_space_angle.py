def group_space_angle_hist(normalized_dfs, pxpermm):
    """Calculate and return a 2D histogram of the angular and distance differences between pairs of flies based on their
      positions, using normalized dataframes."""
    
    degree_bins = np.arange(-177.5, 177.6, settings.ANGLE_BIN)
    distance_bins = np.arange(0.125, 99.8751, settings.DISTANCE_BIN)
    total = np.zeros((len(degree_bins) + 1, len(distance_bins) - 1))

    for fly1_key, fly2_key in list(itertools.permutations(normalized_dfs.keys(), 2)):
        df1, df2 = normalized_dfs[fly1_key].copy(deep=True), normalized_dfs[fly2_key].copy(deep=True)
        df1_array, df2_array = df1.to_numpy(), df2.to_numpy()
        a = np.mean(df1_array[:, 3])
        distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0]) ** 2 + (df1_array[:, 1] - df2_array[:, 1]) ** 2)
        distance = np.round(distance / (a * 4), 4)
        checkang = (np.arctan2(df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])) * 180 / np.pi
        angle = np.round(angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi))

        if settings.MOVECUT:
            movement = np.sqrt(
                (df1_array[:, 0] - np.roll(df1_array[:, 0], 1)) ** 2
                + (df1_array[:, 1] - np.roll(df1_array[:, 1], 1)) ** 2
            )
            movement[0] = movement[1]
            movement = movement / pxpermm[fly1_key] / settings.FPS
            n, c = np.histogram(movement, bins=np.arange(0, 2.51, 0.01))
            peaks, _ = find_peaks((np.max(n) - n) / np.max(np.max(n) - n), prominence=0.05)
            movecut = 0 if len(peaks) == 0 else c[peaks[0]]
            mask = (distance <= settings.DISTANCE_MAX) & (movement > (movecut * pxpermm[fly1_key] / settings.FPS))
        
        else:
            mask = distance <= settings.DISTANCE_MAX

        angle, distance = angle[mask], distance[mask]
        hist, _, _ = np.histogram2d(
            angle,
            distance,
            bins=(degree_bins, distance_bins),
            range=[[-180, 180], [0, 100.0]],
        )

        hist = hist.T
        temp = np.mean([hist[:, 0], hist[:, -1]], axis=0)
        hist = np.hstack((temp[:, np.newaxis], hist, temp[:, np.newaxis]))
        total += hist.T

    norm_total = np.ceil((total / np.max(total)) * 256)
    norm_total = norm_total.T

    return norm_total


def pseudo_group_space_angle_hist(normalized_dfs, pxpermm):
    """ """
    degree_bins = np.arange(-177.5, 177.6, settings.ANGLE_BIN)
    distance_bins = np.arange(0.125, 99.8751, settings.DISTANCE_BIN)
    total = np.zeros((len(degree_bins) + 1, len(distance_bins) - 1))

    for fly1_key, fly2_key in list(itertools.permutations(normalized_dfs.keys(), 2)):
        df1, df2 = normalized_dfs[fly1_key].copy(deep=True), normalized_dfs[fly2_key].copy(deep=True)
        df1_array, df2_array = df1.to_numpy(), df2.to_numpy()
        a = np.mean(df1_array[:, 3])
        distance = np.sqrt((df1_array[:, 0] - df2_array[:, 0]) ** 2 + (df1_array[:, 1] - df2_array[:, 1]) ** 2)
        distance = np.round(distance / (a * 4), 4)
        checkang = (np.arctan2(df2_array[:, 1] - df1_array[:, 1], df2_array[:, 0] - df1_array[:, 0])) * 180 / np.pi
        angle = np.round(angledifference_nd(checkang, df1_array[:, 2] * 180 / np.pi))

        if settings.MOVECUT:
            movement = np.sqrt(
                (df1_array[:, 0] - np.roll(df1_array[:, 0], 1)) ** 2
                + (df1_array[:, 1] - np.roll(df1_array[:, 1], 1)) ** 2
            )
            movement[0] = movement[1]
            movement = movement / pxpermm[fly1_key] / settings.FPS
            n, c = np.histogram(movement, bins=np.arange(0, 2.51, 0.01))
            peaks, _ = find_peaks((np.max(n) - n) / np.max(np.max(n) - n), prominence=0.05)
            movecut = 0 if len(peaks) == 0 else c[peaks[0]]
            mask = (distance <= settings.DISTANCE_MAX) & (movement > (movecut * pxpermm[fly1_key] / settings.FPS))

        else:
            mask = distance <= settings.DISTANCE_MAX

        angle, distance = angle[mask], distance[mask]
        hist, _, _ = np.histogram2d(
            angle,
            distance,
            bins=(degree_bins, distance_bins),
            range=[[-180, 180], [0, 100.0]],
        )

        hist = hist.T
        temp = np.mean([hist[:, 0], hist[:, -1]], axis=0)
        hist = np.hstack((temp[:, np.newaxis], hist, temp[:, np.newaxis]))
        total += hist.T

    total = np.ceil(total)

    return total.T