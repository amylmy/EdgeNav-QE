# Adaptive threshold based on image blur score and text instruction length
def adaptive_threshold(image, text):
    blur_score = calculate_blur(image)  # Custom image blur calculation
    text_len = len(text.split())
    if blur_score > 0.5 or text_len > 8:
        return 0.98  # Higher threshold for complex samples (reduce early exit)
    else:
        return 0.90  # Lower threshold for simple samples (prioritize early exit)

# Apply adaptive threshold during inference
finetuned_model.exit_threshold = adaptive_threshold(image, "pick up the red cube")