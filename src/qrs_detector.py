"""
QRS detection - Python port of CommwellSentinelAnalysis.QRS() and
thresholdDetector() from Stream-Analysis-Service.

Processes 200 Hz signal (after preprocessing).
"""

import numpy as np
from typing import Tuple
from scipy.signal import butter, sosfilt


# QRS bandpass filter coefficients from C# (5 Hz HP + 20 Hz LP at 200 Hz)
# Port of: HP5hz and LP20hz filters used in QRS() method
_bHP5 = [0.664988107305248, -3.324940536526238, 6.649881073052476,
         -6.649881073052476, 3.324940536526238, -0.664988107305248]
_aHP5 = [1.0, -4.187300047864400, 7.069722752792471,
         -6.009958148187332, 2.570429302524102, -0.442209182399620]
_bLP20 = [4.89436086134670e-004, 2.44718043067335e-003, 4.89436086134670e-003,
          4.89436086134670e-003, 2.44718043067335e-003, 4.89436086134670e-004]
_aLP20 = [1.0, -3.378011394596868, 4.751775374707016,
          -3.439713310835590, 1.273999881482925, -0.192388596001173]


def _qrs_filter(signal: np.ndarray) -> np.ndarray:
    """
    QRS detection filter: 5-20 Hz bandpass + derivative + abs + window sum.
    Matches C# QRS() method.
    """
    n = len(signal)
    # Use scipy SOS for the bandpass instead of sample-by-sample IIR
    sos_hp = butter(3, 5, btype='highpass', fs=200, output='sos')
    sos_lp = butter(3, 20, btype='lowpass', fs=200, output='sos')

    filtered = sosfilt(sos_hp, signal.astype(np.float64))
    filtered = sosfilt(sos_lp, filtered) * 10.0

    # Derivative: abs(f[i+1] - f[i])
    deriv = np.zeros(n, dtype=np.float64)
    deriv[:-1] = np.abs(np.diff(filtered))

    # Sliding window sum (16 samples) - matches C# inner loop
    window = 16
    cumsum = np.cumsum(deriv)
    windowed = np.zeros(n, dtype=np.float64)
    windowed[window:] = cumsum[window:] - cumsum[:-window]

    return windowed


def detect_qrs(signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    Detect QRS complexes using threshold detection.
    Port of C# thresholdDetector().

    Args:
        signal: preprocessed ECG signal at fs Hz
        fs: sample rate (200 Hz)

    Returns:
        markers: Nx2 array where col 0 = sample position, col 1 = beat type
                 beat type: 1 = normal, 5 = ventricular, 0 = noise
    """
    deriv = _qrs_filter(signal)
    n = len(deriv)

    markers = []
    threshold = 100.0
    last_hold = 100.0
    peak_value = 0.0
    peak_time = 0
    found = False
    peaks_sum = 0.0
    peak_count = 0
    no_beat_count = 0
    long_term_peak = 0.0
    long_term_cnt = 0
    retest = False

    i = 40
    while i < n - 16:
        if deriv[i] > threshold and i < peak_time + 40:
            peak_time = i
            peak_value = deriv[i]
            threshold = peak_value
            found = True
            no_beat_count = 0

        if not found:
            peak_time = i
            no_beat_count += 1
            if no_beat_count > 600:  # 3 sec without beats
                if not retest:
                    i = max(40, i - no_beat_count)
                    retest = True
                last_hold = 100.0
                no_beat_count = 0
                threshold = 100.0

        elif i > peak_time + 40 and found:
            i = peak_time + 50
            found = False
            retest = False

            # Find true peak in ±40 window
            true_peak = 0.0
            true_peak_time = peak_time
            search_start = max(0, peak_time - 40)
            for j in range(peak_time, search_start, -1):
                if j < n and deriv[j] > true_peak:
                    true_peak = deriv[j]
                    true_peak_time = j

            # Only count peaks > 0.15 mV threshold (10 digital units)
            if true_peak_time < n and deriv[true_peak_time] > 10:
                pos = true_peak_time if true_peak != 0 else peak_time
                beat_type = 1 if peak_value < 1500 else 0
                markers.append([pos, beat_type])

            peaks_sum += peak_value
            peak_count += 1
            long_term_peak += peak_value
            long_term_cnt += 1

            if peak_count == 8:
                last_hold = peaks_sum / 24.0  # 1/3 average peak
                peak_count = 0
                peaks_sum = 0
                if long_term_cnt > 0:
                    lt_avg = long_term_peak / long_term_cnt / 3
                    if last_hold > lt_avg:
                        last_hold = lt_avg
            threshold = last_hold

        i += 1

    if not markers:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(markers, dtype=np.int32)


def compute_heart_rate(markers: np.ndarray, fs: int = 200) -> dict:
    """
    Compute heart rate statistics from QRS markers.
    Port of C# HeartRateStatistics().

    Returns:
        dict with hr_mean, hr_std, hr_min, hr_max, rr_intervals
    """
    if len(markers) < 2:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    # Filter to normal beats only
    normal_mask = markers[:, 1] == 1
    normal_positions = markers[normal_mask, 0]

    if len(normal_positions) < 2:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    rr_intervals = np.diff(normal_positions)
    rr_intervals = rr_intervals[rr_intervals > 0]

    if len(rr_intervals) == 0:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    hr_values = (fs * 60.0) / rr_intervals
    # Filter unrealistic values
    valid = (hr_values > 20) & (hr_values < 300)
    hr_values = hr_values[valid]
    rr_intervals = rr_intervals[valid]

    if len(hr_values) == 0:
        return {"hr_mean": 0, "hr_std": 0, "hr_min": 0, "hr_max": 0, "rr_intervals": []}

    return {
        "hr_mean": float(np.mean(hr_values)),
        "hr_std": float(np.std(hr_values)),
        "hr_min": float(np.min(hr_values)),
        "hr_max": float(np.max(hr_values)),
        "rr_intervals": rr_intervals.tolist(),
    }


