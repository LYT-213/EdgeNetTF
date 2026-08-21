import numpy as np
from scipy import stats


def causal_majority_vote(predictions, window_size=5):
    """Causal majority vote using the current and up to window_size-1 prior predictions."""
    predictions = np.asarray(predictions)
    smoothed = np.copy(predictions)

    for i in range(len(predictions)):
        start = max(0, i - window_size + 1)
        smoothed[i] = stats.mode(
            predictions[start:i + 1],
            keepdims=False,
        )[0]

    return smoothed


def segmented_causal_majority_vote(predictions, segment_ids, window_size=5):
    """Apply causal voting independently inside dataset-defined gesture segments."""
    predictions = np.asarray(predictions)
    segment_ids = np.asarray(segment_ids)
    smoothed = np.copy(predictions)

    for segment_id in np.unique(segment_ids):
        mask = segment_ids == segment_id
        smoothed[mask] = causal_majority_vote(
            predictions[mask],
            window_size=window_size,
        )

    return smoothed
