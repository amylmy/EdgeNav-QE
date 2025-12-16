# Calibrate DEE thresholds for navigation
def calibrate_nav_thresholds(calib_df):
    # Define navigation targets
    max_heading_error = 2.0  # Degrees
    min_path_completion = 0.95
    max_collision_rate = 0.01
    
    # Calibrate layer 4 threshold
    thresholds_4 = np.linspace(0.85, 0.98, 15)
    optimal_thresh_4 = 0.92  # Default
    for t in thresholds_4:
        valid = calib_df[
            (calib_df["conf_4"] >= t) &
            (calib_df["heading_error"] <= max_heading_error) &
            (calib_df["path_completion"] >= min_path_completion) &
            (calib_df["collision_rate"] <= max_collision_rate)
        ]
        if len(valid) / len(calib_df) >= 0.4:  # 40% early exit rate
            optimal_thresh_4 = t
            break
    
    # Calibrate layer 8 threshold (higher for stricter navigation)
    thresholds_8 = np.linspace(0.90, 0.99, 15)
    optimal_thresh_8 = 0.95  # Default
    for t in thresholds_8:
        valid = calib_df[
            (calib_df["conf_8"] >= t) &
            (calib_df["heading_error"] <= 1.0) &  # Stricter for layer 8
            (calib_df["path_completion"] >= 0.98)
        ]
        if len(valid) / len(calib_df) >= 0.7:  # 70% early exit rate
            optimal_thresh_8 = t
            break
    
    return optimal_thresh_4, optimal_thresh_8