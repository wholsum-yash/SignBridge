import numpy as np

# THRESHOLDS (TO FILTER OUT GOOD DATA)
VALID_THRESHOLD = 0.5
DIVERSITY_THRESHOLD = 0.2
FALLBACK_THRESHOLD = 0.6

BASE_FEATURES = 126

# threshold computation
def compute_metrics(sequence, fallback_ratio):
    """
    Computes quality metrics for the sequence.

    [IMPORTANT]: Added fallback_ratio to the metrics dictionary so it can be returned and used for validation.
    """
    total_frames = len(sequence)

    if total_frames == 0:
        return{
            "valid_ratio" : 0,
            "diversity_ratio" : 0,
            "fallback_ratio" : fallback_ratio
        }

    landmarks = sequence[:, :BASE_FEATURES]
    # velocity = sequence[:, :BASE_FEATURES] NOT USING FOR VALIDATION

    # valid frames
    non_zero_frames = np.sum(np.any(landmarks != 0, axis = 1))
    valid_ratio = non_zero_frames / total_frames

    # unique frames
    unique_frames = len(np.unique(landmarks, axis = 0))
    diversity_ratio = unique_frames / total_frames

    return {
        "valid_ratio" : valid_ratio,
        "diversity_ratio" : diversity_ratio,
        "fallback_ratio": fallback_ratio # Added to allow validation check in validate_sequence
    }

# validation function
def validate_sequence(sequence, fallback_ratio):
    """ 
    INPUT: 
        sequence -> np.array (the repaired sequence)
        fallback_ratio -> float (ratio of frames that were repaired)
    OUTPUT: 
        is-valid -> bool; reason: str; metrics: dict 
        
    'fallback_ratio' is passed here to ensure we don't use sequences that required too much repair.
    """

    metrics = compute_metrics(sequence, fallback_ratio)

    if metrics["valid_ratio"] < VALID_THRESHOLD:
        return False, "low_valid", metrics

    if metrics["diversity_ratio"] < DIVERSITY_THRESHOLD:
        return False, "low_diversity", metrics
    
    # Check if the sequence has too many synthetic/copied frames
    if metrics["fallback_ratio"] > FALLBACK_THRESHOLD:
        return False, "too_repaired", metrics # Fixed typo 'too_reparied'
   
    return True, "pass", metrics 
