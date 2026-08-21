import numpy as np


def compress_semg(emg):
    """Logarithmic amplitude compression used in the original experiments."""
    emg = np.asarray(emg)
    return np.sign(emg) * np.log(1.0 + 2048.0 * np.abs(emg)) / np.log(2049.0)


def fft_representation(time_window):
    """Return the log-transformed first half of the FFT magnitude spectrum.

    Expected input shape: [time, channels], with 600 samples in the manuscript.
    Output shape for a 600-sample window: [300, channels].
    """
    time_window = np.asarray(time_window)
    half = time_window.shape[0] // 2
    spectrum = np.abs(np.fft.fft(time_window, axis=0))[:half, :]
    return np.log1p(spectrum + 1e-8)


def standardize_train_test(train, test):
    """Standardize using training-window statistics only."""
    mean = np.mean(train, axis=(0, 1), keepdims=True)
    std = np.std(train, axis=(0, 1), keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


def create_segmented_windows(
    emg,
    labels,
    repetitions,
    target_repetitions,
    window_size=600,
    stride=60,
    valid_label_min=1,
    valid_label_max=49,
):
    """Create windows without crossing repetition or gesture-segment boundaries.

    Returns temporal windows, FFT windows, labels, and segment IDs.
    """
    X_t, X_f, y, segment_ids = [], [], [], []
    segment_id = 0

    labels = np.asarray(labels).reshape(-1)
    repetitions = np.asarray(repetitions).reshape(-1)

    for rep in target_repetitions:
        idx = np.where(repetitions == rep)[0]
        if len(idx) < window_size:
            continue

        continuous_blocks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

        for block in continuous_blocks:
            if len(block) < window_size:
                continue

            block_labels = labels[block].astype(np.int64)
            change_points = np.where(np.diff(block_labels) != 0)[0] + 1

            for label_segment in np.split(block, change_points):
                if len(label_segment) < window_size:
                    continue

                label = int(labels[label_segment[0]])
                current_segment_id = segment_id
                segment_id += 1

                if not (valid_label_min <= label <= valid_label_max):
                    continue

                for i in range(0, len(label_segment) - window_size + 1, stride):
                    win_idx = label_segment[i:i + window_size]
                    if int(labels[win_idx[-1]]) != label:
                        continue

                    t_win = emg[win_idx]
                    X_t.append(t_win)
                    X_f.append(fft_representation(t_win))
                    y.append(label)
                    segment_ids.append(current_segment_id)

    return (
        np.asarray(X_t, dtype=np.float32),
        np.asarray(X_f, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(segment_ids, dtype=np.int64),
    )
