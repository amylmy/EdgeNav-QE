'''
Advanced: Adaptive Thresholds (Per-Sample/Per-Complexity)
For better robustness, replace fixed thresholds with adaptive thresholds that adjust based on sample complexity.
Per-Complexity Thresholds:
Set different thresholds for simple/medium/complex samples (labeled during calibration).
'''

def adaptive_threshold(sample_complexity, layer_idx):
    # Pre-calibrated thresholds per complexity/layer
    thresholds = {
        "simple": {4: 0.88, 8: 0.92},
        "medium": {4: 0.92, 8: 0.95},
        "complex": {4: 0.98, 8: 0.99}  # High thresholds (rarely exit early)
    }
    return thresholds[sample_complexity][layer_idx]

# Usage during inference
sample_complexity = predict_complexity(image, text)  # Custom complexity classifier
model.exit_threshold = adaptive_threshold(sample_complexity, current_layer_idx)

'''
 Dynamic Thresholds (Online Calibration)
 Update thresholds during deployment based on real-world performance
'''
class DynamicThresholdTracker:
    def __init__(self, initial_thresh_4=0.92, initial_thresh_8=0.95):
        self.thresh_4 = initial_thresh_4
        self.thresh_8 = initial_thresh_8
        self.error_buffer = {"4": [], "8": []}  # Track MAE of early-exit samples
    
    def update_thresholds(self, layer_idx, mae):
        # Add new MAE to buffer
        self.error_buffer[str(layer_idx)].append(mae)
        # Keep buffer size to 100 samples (recent performance)
        if len(self.error_buffer[str(layer_idx)]) > 100:
            self.error_buffer[str(layer_idx)].pop(0)
        
        # Adjust threshold if average MAE exceeds target
        avg_mae = np.mean(self.error_buffer[str(layer_idx)])
        target_mae = 0.05  # Example target
        if avg_mae > target_mae:
            # Increase threshold by 0.01 to reduce early exits (improve accuracy)
            if layer_idx == 4:
                self.thresh_4 = min(self.thresh_4 + 0.01, 0.99)
            else:
                self.thresh_8 = min(self.thresh_8 + 0.01, 0.99)
        else:
            # Decrease threshold by 0.01 to increase early exits (improve efficiency)
            if layer_idx == 4:
                self.thresh_4 = max(self.thresh_4 - 0.01, 0.80)
            else:
                self.thresh_8 = max(self.thresh_8 - 0.01, 0.85)

# Usage
tracker = DynamicThresholdTracker()
# After each inference:
if exit_layer in [4,8]:
    tracker.update_thresholds(exit_layer, current_mae)
# Apply updated thresholds
model.exit_thresholds = {4: tracker.thresh_4, 8: tracker.thresh_8}

