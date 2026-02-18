"""
ECG Quality Analysis for Sentinel/Holter 2-lead recordings.
Adapted from Ecg-Interpretation-Python-Service quality.py
with HOLTER_THRESHOLDS for 200 Hz 2-lead signals.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

logger = logging.getLogger(__name__)

# Holter-specific thresholds for 200 Hz 2-lead recordings
HOLTER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Muscle_Artifact": (0.045, 0.10),
    "Bad_Electrode_Contact": (10, 800),
    "Powerline_Interference": (0.01, 0.05),
    "Baseline_Drift": (0.03, 0.90),
    "Low_SNR": (22, 12),
}

FLAGS_WEIGHTS: Dict[str, float] = {
    "Muscle_Artifact": 0.2,
    "Bad_Electrode_Contact": 0.25,
    "Powerline_Interference": 0.15,
    "Baseline_Drift": 0.2,
    "Low_SNR": 0.2,
}

FLAG_MESSAGES: Dict[str, str] = {
    "Muscle_Artifact": "Excess muscle noise",
    "Bad_Electrode_Contact": "Poor electrode contact",
    "Powerline_Interference": "Power-line interference detected",
    "Baseline_Drift": "Baseline drift present",
    "Low_SNR": "Low signal-to-noise ratio",
}

_FALLBACK_CONFIG: Dict[str, Any] = {
    "_preset_name": "hardcoded_default",
    "thresholds": dict(HOLTER_THRESHOLDS),
    "flags_weights": dict(FLAGS_WEIGHTS),
    "neurokit": {"enabled": False, "method": "averageQRS", "weight": 0.0},
    "grade_thresholds": {"good": 0.85, "questionable": 0.65},
}


def load_quality_config(preset: str = None) -> Dict[str, Any]:
    """Load quality config for a single preset. Falls back to hardcoded defaults on any error."""
    presets_dir = _get_presets_dir()
    try:
        if preset is None:
            meta = _load_meta()
            preset = meta.get("default", "holter_200hz")
        preset_path = presets_dir / f"{preset}.json"
        preset_data = json.loads(preset_path.read_text(encoding="utf-8"))
        # Deep merge: fallback provides defaults, preset overrides per-key at each level
        merged = copy.deepcopy(_FALLBACK_CONFIG)
        for key, value in preset_data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
        merged["_preset_name"] = preset
        return merged
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load quality config (%s), using fallback defaults", exc)
        return copy.deepcopy(_FALLBACK_CONFIG)


def _get_presets_dir() -> Path:
    """Return the path to the presets directory."""
    return Path(__file__).resolve().parent.parent / "config" / "presets"


def _load_meta() -> Dict[str, Any]:
    """Load _meta.json from presets directory."""
    meta_path = _get_presets_dir() / "_meta.json"
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"default": "holter_200hz"}


def load_all_presets() -> Dict[str, Any]:
    """Load all presets from individual JSON files in the presets directory.
    Returns the same structure as before: {"default": "...", "presets": {...}}.
    On error, returns a minimal structure with _FALLBACK_CONFIG as the single preset."""
    presets_dir = _get_presets_dir()
    try:
        meta = _load_meta()
        presets = {}
        for preset_file in sorted(presets_dir.glob("*.json")):
            if preset_file.name.startswith("_"):
                continue
            name = preset_file.stem
            presets[name] = json.loads(preset_file.read_text(encoding="utf-8"))
        if not presets:
            raise ValueError("No preset files found in config/presets/")
        return {"default": meta.get("default", "holter_200hz"), "presets": presets}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Could not load presets (%s), using fallback", exc)
        return {
            "default": "hardcoded_default",
            "presets": {"hardcoded_default": copy.deepcopy(_FALLBACK_CONFIG)},
        }


def save_preset(name: str, preset_data: Dict[str, Any]) -> None:
    """Write a single preset to its own JSON file. Uses write-to-temp + rename for safety."""
    presets_dir = _get_presets_dir()
    presets_dir.mkdir(parents=True, exist_ok=True)
    preset_path = presets_dir / f"{name}.json"
    tmp_path = preset_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(preset_data, indent=2), encoding="utf-8")
        tmp_path.replace(preset_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_presets(data: Dict[str, Any]) -> None:
    """Write all presets from the full structure to individual files.
    Also updates _meta.json with the default preset name."""
    presets_dir = _get_presets_dir()
    presets_dir.mkdir(parents=True, exist_ok=True)
    # Save meta
    meta = {k: v for k, v in data.items() if k != "presets"}
    meta_path = presets_dir / "_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Save each preset as individual file
    for name, preset_data in data.get("presets", {}).items():
        save_preset(name, preset_data)


def analyze_lead_quality(
    signal: np.ndarray,
    sampling_rate: int = 200,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Compute PSD-based signal-quality metrics for a single ECG segment.
    NeuroKit2 quality is computed separately on the full signal in analyze_holter_quality().
    """
    t_grades = thresholds if thresholds is not None else HOLTER_THRESHOLDS

    sig = signal - np.mean(signal)
    freqs, psd = welch(sig, fs=sampling_rate)
    total_power = np.sum(psd) + 1e-12

    flags: Dict[str, float] = {}

    # Notch 50/60 Hz before analysis
    b50, a50 = iirnotch(50, Q=50, fs=sampling_rate)
    sos50 = tf2sos(b50, a50)
    b60, a60 = iirnotch(60, Q=50, fs=sampling_rate)
    sos60 = tf2sos(b60, a60)

    sig_notched = sosfiltfilt(sos50, sig)
    sig_notched = sosfiltfilt(sos60, sig_notched)

    nperseg = min(int(2.5 * sampling_rate), len(sig_notched))
    if nperseg < 4:
        nperseg = len(sig_notched)
    freqs2, psd2 = welch(sig_notched, fs=sampling_rate, nperseg=nperseg)
    total_power2 = np.sum(psd2) + 1e-12

    # Powerline interference (48-52 and 58-62 Hz)
    mains_bins = ((freqs2 >= 48) & (freqs2 <= 52)) | ((freqs2 >= 58) & (freqs2 <= 62))
    mains_power = np.sum(psd2[mains_bins])
    pi = mains_power / total_power2
    flags["Powerline_Interference"] = float(
        np.clip(
            (pi - t_grades["Powerline_Interference"][0])
            / (
                t_grades["Powerline_Interference"][1]
                - t_grades["Powerline_Interference"][0]
            ),
            0,
            1,
        )
    )

    # Muscle artifact (35-100 Hz)
    hf_bins = (freqs2 >= 35) & (freqs2 <= 100)
    hf_power = np.sum(psd2[hf_bins])
    ma_ratio = hf_power / total_power2
    flags["Muscle_Artifact"] = float(
        np.clip(
            (ma_ratio - t_grades["Muscle_Artifact"][0])
            / (t_grades["Muscle_Artifact"][1] - t_grades["Muscle_Artifact"][0]),
            0,
            1,
        )
    )

    # Baseline drift (<0.5 Hz)
    lf = np.sum(psd[freqs < 0.5])
    bd = lf / total_power
    flags["Baseline_Drift"] = float(
        np.clip(
            (bd - t_grades["Baseline_Drift"][0])
            / (t_grades["Baseline_Drift"][1] - t_grades["Baseline_Drift"][0]),
            0,
            1,
        )
    )

    # QRS amplitude
    amp = float(np.ptp(sig))

    # Bad electrode contact
    if amp < t_grades["Bad_Electrode_Contact"][0]:
        flags["Bad_Electrode_Contact"] = 1.0
    elif amp > t_grades["Bad_Electrode_Contact"][1]:
        flags["Bad_Electrode_Contact"] = 1.0
    else:
        flags["Bad_Electrode_Contact"] = 0.0

    # SNR
    sos_hp = butter(4, 1, btype="highpass", fs=sampling_rate, output="sos")
    clean_hp = sosfiltfilt(sos_hp, sig)
    amplitude = clean_hp.max() - clean_hp.min()

    if amplitude < 5:
        # Flat-line signal — unconditionally worst SNR score
        flags["Low_SNR"] = 1.0
        snr = 0.1
    else:
        try:
            nyq = sampling_rate / 2
            hi = min(40, nyq - 1)
            sos = butter(2, [0.5, hi], btype="bandpass", fs=sampling_rate, output="sos")
            clean = sosfiltfilt(sos, sig)
            noise = sig - clean
            noise_power = np.mean(noise**2) + 1e-12
            snr = 10 * np.log10((amp**2) / noise_power)
        except Exception:
            signal_power = np.mean(sig**2) + 1e-12
            noise_power = np.var(sig) + 1e-12
            snr = 10 * np.log10(signal_power / noise_power)

        flags["Low_SNR"] = float(
            np.clip(
                (snr - t_grades["Low_SNR"][0])
                / (t_grades["Low_SNR"][1] - t_grades["Low_SNR"][0]),
                0,
                1,
            )
        )

    return {
        "flags": flags,
        "values": {
            "m_a": ma_ratio,
            "b_e_c": amp,
            "p_i": pi,
            "b_d": bd,
            "snr": snr,
            "qrs_amp": amp,
        },
    }


def compute_quality_score(
    flags: Dict[str, float],
    nk_quality: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Derive quality score (0.0-1.0) where each flag deducts weighted penalty.
    Optionally blends with NeuroKit quality index when available."""
    weights = config["flags_weights"] if config and "flags_weights" in config else FLAGS_WEIGHTS
    psd_score = 1.0
    for flag, value in flags.items():
        if value > 0.0:
            psd_score -= weights.get(flag, 0.2) * value
    psd_score = max(0.0, min(1.0, psd_score))

    if nk_quality is not None and config and config.get("neurokit", {}).get("enabled", False):
        nk_weight = config["neurokit"]["weight"]
        score = psd_score * (1.0 - nk_weight) + nk_quality * nk_weight
    else:
        score = psd_score

    return max(0.0, min(1.0, score))


def analyze_holter_quality(
    lead1: np.ndarray,
    lead2: np.ndarray,
    sampling_rate: int = 200,
    window_sec: float = 5,
    preset: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze quality of 2-lead Holter recording.
    Segments into windows, finds best window (~5 sec) for PSD metrics.
    NeuroKit2 quality runs on the full signal and blends with PSD at the end.
    Uses best-lead selection (max) instead of weighted average.

    When `config` is provided, uses it directly instead of loading from file.
    When `config` is None, falls back to `load_quality_config(preset)`.
    """
    if config is None:
        config = load_quality_config(preset)
    wlen = int(window_sec * sampling_rate)
    n = min(len(lead1), len(lead2))
    nwin = n // wlen if wlen > 0 else 0

    if nwin == 0:
        _empty_values = {"m_a": 0.0, "b_e_c": 0.0, "p_i": 0.0, "b_d": 0.0, "snr": 0.0, "qrs_amp": 0.0, "nk_quality": None, "nk_r_peaks_count": 0}
        return {
            "lead1_quality": 0.0,
            "lead2_quality": 0.0,
            "lead1_psd_quality": 0.0,
            "lead2_psd_quality": 0.0,
            "overall_quality": 0.0,
            "grade": "Not usable",
            "lead1_flags": {},
            "lead2_flags": {},
            "lead1_values": dict(_empty_values),
            "lead2_values": dict(_empty_values),
            "window_scores": [],
            "best_window_start": 0,
            "best_window_end": 0,
            "quality_best_lead": 1,
            "lead1_nk_quality": None,
            "lead2_nk_quality": None,
            "preset": config.get("_preset_name", "default"),
        }

    window_results = []
    for i in range(nwin):
        seg1 = lead1[i * wlen : (i + 1) * wlen]
        seg2 = lead2[i * wlen : (i + 1) * wlen]

        m1 = analyze_lead_quality(seg1, sampling_rate, thresholds=config["thresholds"])
        m2 = analyze_lead_quality(seg2, sampling_rate, thresholds=config["thresholds"])

        q1 = compute_quality_score(m1["flags"], config=config)
        q2 = compute_quality_score(m2["flags"], config=config)
        overall = max(q1, q2)

        window_results.append(
            {
                "window": i,
                "lead1_score": q1,
                "lead2_score": q2,
                "overall": overall,
                "lead1_flags": m1["flags"],
                "lead2_flags": m2["flags"],
                "lead1_values": m1["values"],
                "lead2_values": m2["values"],
            }
        )

    # Find best consecutive window(s) (~5 sec total)
    best_start = 0
    best_quality = -1.0
    windows_needed = max(1, int(5.0 / window_sec))
    for start in range(max(1, nwin - windows_needed + 1)):
        seq = window_results[start : start + windows_needed]
        avg_q = np.mean([w["overall"] for w in seq])
        if avg_q > best_quality:
            best_quality = avg_q
            best_start = start

    best_windows = window_results[best_start : best_start + windows_needed]
    q1_psd_avg = float(np.mean([w["lead1_score"] for w in best_windows]))
    q2_psd_avg = float(np.mean([w["lead2_score"] for w in best_windows]))

    # Aggregate flags from best windows
    agg_flags1 = {}
    agg_flags2 = {}
    for flag in FLAG_MESSAGES:
        vals1 = [w["lead1_flags"].get(flag, 0) for w in best_windows]
        vals2 = [w["lead2_flags"].get(flag, 0) for w in best_windows]
        agg_flags1[flag] = float(np.mean(vals1))
        agg_flags2[flag] = float(np.mean(vals2))

    # Aggregate raw measurement values from best windows
    _value_keys = ["m_a", "b_e_c", "p_i", "b_d", "snr", "qrs_amp"]
    lead1_values_agg = {}
    lead2_values_agg = {}
    for key in _value_keys:
        lead1_values_agg[key] = float(np.mean([w["lead1_values"][key] for w in best_windows]))
        lead2_values_agg[key] = float(np.mean([w["lead2_values"][key] for w in best_windows]))

    # NeuroKit2 quality on full signals (not windowed)
    nk_cfg = config.get("neurokit", {})
    lead1_nk_quality = None
    lead2_nk_quality = None
    lead1_nk_r_peaks = 0
    lead2_nk_r_peaks = 0
    if nk_cfg.get("enabled", False):
        nk_min_samples = sampling_rate * 4  # ecg_segment() minimum
        for lead_sig, lead_num in [(lead1[:n], 1), (lead2[:n], 2)]:
            if len(lead_sig) < nk_min_samples:
                logger.debug(
                    "Lead %d too short for NeuroKit2 (%d < %d), skipping",
                    lead_num, len(lead_sig), nk_min_samples,
                )
                continue
            try:
                import neurokit2 as nk

                cleaned = nk.ecg_clean(lead_sig, sampling_rate=sampling_rate)
                _, peaks_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
                r_peaks = list(peaks_info.get("ECG_R_Peaks", []))
                quality_arr = nk.ecg_quality(
                    cleaned,
                    rpeaks=peaks_info.get("ECG_R_Peaks"),
                    sampling_rate=sampling_rate,
                    method=nk_cfg.get("method", "averageQRS"),
                )
                quality_arr = np.asarray(quality_arr, dtype=float)
                nk_q = float(np.clip(np.nanmean(quality_arr), 0.0, 1.0))
                if lead_num == 1:
                    lead1_nk_quality = nk_q
                    lead1_nk_r_peaks = len(r_peaks)
                else:
                    lead2_nk_quality = nk_q
                    lead2_nk_r_peaks = len(r_peaks)
            except Exception as exc:
                logger.warning("NeuroKit2 quality computation failed for lead %d: %s", lead_num, exc)

    # Include NK fields in aggregated values
    lead1_values_agg["nk_quality"] = lead1_nk_quality
    lead2_values_agg["nk_quality"] = lead2_nk_quality
    lead1_values_agg["nk_r_peaks_count"] = lead1_nk_r_peaks
    lead2_values_agg["nk_r_peaks_count"] = lead2_nk_r_peaks

    # Blend PSD (best-window) with NK (whole-signal) for final scores
    nk_weight = nk_cfg.get("weight", 0.0) if nk_cfg.get("enabled", False) else 0.0
    q1_avg = q1_psd_avg
    q2_avg = q2_psd_avg
    if lead1_nk_quality is not None and nk_weight > 0:
        q1_avg = q1_psd_avg * (1.0 - nk_weight) + lead1_nk_quality * nk_weight
    if lead2_nk_quality is not None and nk_weight > 0:
        q2_avg = q2_psd_avg * (1.0 - nk_weight) + lead2_nk_quality * nk_weight
    overall = max(q1_avg, q2_avg)

    grade_thresholds = config["grade_thresholds"]
    if overall > grade_thresholds["good"]:
        grade = "Good"
    elif overall > grade_thresholds["questionable"]:
        grade = "Questionable"
    else:
        grade = "Not usable"

    return {
        "lead1_quality": q1_avg,
        "lead2_quality": q2_avg,
        "lead1_psd_quality": q1_psd_avg,
        "lead2_psd_quality": q2_psd_avg,
        "overall_quality": overall,
        "grade": grade,
        "lead1_flags": agg_flags1,
        "lead2_flags": agg_flags2,
        "lead1_values": lead1_values_agg,
        "lead2_values": lead2_values_agg,
        "best_window_start": best_start,
        "best_window_end": best_start + windows_needed,
        "quality_best_lead": 1 if q1_avg >= q2_avg else 2,
        "lead1_nk_quality": lead1_nk_quality,
        "lead2_nk_quality": lead2_nk_quality,
        "preset": config.get("_preset_name", "default"),
        "window_scores": [
            {
                "window": w["window"],
                "lead1": w["lead1_score"],
                "lead2": w["lead2_score"],
                "overall": w["overall"],
            }
            for w in window_results
        ],
    }
