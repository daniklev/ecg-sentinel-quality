"""
Config optimizer for ECG Sentinel Quality.

Uses scipy.optimize.differential_evolution to search the 20-parameter
quality-config space, minimizing loss between actual and expected grades
across labeled test recordings.
"""

import glob
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from src.dat_parser import parse_dat_file
from src.filters import preprocess_dat_signal
from src.quality import analyze_holter_quality

# NeuroKit method options (index → name)
_NK_METHODS = ["averageQRS", "zhao2018", "orphanidou2015"]

# Grade label → target score for loss computation
GRADE_TARGETS = {
    "Good": 0.95,
    "Questionable": 0.75,
    "Not usable": 0.30,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedFile:
    """Cached parse + filter results for a single .dat file."""
    filename: str
    folder: str
    lead1: np.ndarray
    lead2: np.ndarray


@dataclass
class OptimizationResult:
    """Returned by run_optimization()."""
    best_config: Dict[str, Any]
    best_loss: float
    before_results: List[Dict[str, Any]]
    after_results: List[Dict[str, Any]]
    convergence_history: List[float]
    n_evaluations: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Preprocessing (expensive, config-independent)
# ---------------------------------------------------------------------------

def preprocess_test_data(
    base_folder: str,
    folder_grades: Dict[str, str],
    notch_freqs: List[int],
) -> List[PreprocessedFile]:
    """Parse + filter + resample all .dat files in the labeled folders.

    Args:
        base_folder: root data directory (e.g. "data")
        folder_grades: {folder_name: grade_label} for folders to include
        notch_freqs: notch filter frequencies (e.g. [50, 100])

    Returns:
        List of PreprocessedFile with filtered lead arrays at 200 Hz.
    """
    results: List[PreprocessedFile] = []
    for folder_name in sorted(folder_grades.keys()):
        folder_path = os.path.join(base_folder, folder_name)
        dat_files = sorted(glob.glob(os.path.join(folder_path, "*.dat")))
        for fpath in dat_files:
            raw = open(fpath, "rb").read()
            lead1_raw, lead2_raw = parse_dat_file(raw)
            lead1 = preprocess_dat_signal(lead1_raw, notch_freqs=notch_freqs)
            lead2 = preprocess_dat_signal(lead2_raw, notch_freqs=notch_freqs)
            results.append(PreprocessedFile(
                filename=os.path.basename(fpath),
                folder=folder_name,
                lead1=lead1,
                lead2=lead2,
            ))
    return results


# ---------------------------------------------------------------------------
# Parameter vector encoding / decoding (20 elements)
# ---------------------------------------------------------------------------
# Idx  Param                          Bounds
# 0-1  Muscle_Artifact [low, high]    [0, 1]
# 2-3  Bad_Electrode_Contact [low, high]  [0, 2000]
# 4-5  Powerline_Interference [low, high]  [0, 1]
# 6-7  Baseline_Drift [low, high]     [0, 1]
# 8-9  Low_SNR [good, bad] (inverted) [0, 50]
# 10-14 5 flag weights                [0, 1]
# 15-16 grade_good, grade_questionable [0.3-1.0], [0.1-0.9]
# 17   nk_enabled (>=0.5 = True)      [0, 1]
# 18   nk_method (floor → index)      [0, 2.99]
# 19   nk_weight                      [0, 1]

_FLAG_ORDER = [
    "Muscle_Artifact",
    "Bad_Electrode_Contact",
    "Powerline_Interference",
    "Baseline_Drift",
    "Low_SNR",
]


def get_param_bounds(thresh_config: Dict[str, Dict]) -> List[Tuple[float, float]]:
    """Build 20-element bounds list from _THRESH_CONFIG ranges."""
    bounds = []
    for flag in _FLAG_ORDER:
        cfg = thresh_config[flag]
        bounds.append((cfg["min"], cfg["max"]))  # low threshold
        bounds.append((cfg["min"], cfg["max"]))  # high threshold
    # 5 flag weights
    for _ in _FLAG_ORDER:
        bounds.append((0.0, 1.0))
    # grade thresholds
    bounds.append((0.3, 1.0))   # grade_good
    bounds.append((0.1, 0.9))   # grade_questionable
    # neurokit
    bounds.append((0.0, 1.0))   # nk_enabled
    bounds.append((0.0, 2.99))  # nk_method index
    bounds.append((0.0, 1.0))   # nk_weight
    return bounds


def encode_config(config: Dict[str, Any]) -> np.ndarray:
    """Convert a quality config dict into a 20-element vector (seeds DE)."""
    x = np.zeros(20)
    thresholds = config.get("thresholds", {})
    weights = config.get("flags_weights", {})
    for i, flag in enumerate(_FLAG_ORDER):
        bounds = thresholds.get(flag, (0.0, 1.0))
        if isinstance(bounds, (list, tuple)):
            x[i * 2] = bounds[0]
            x[i * 2 + 1] = bounds[1]
        else:
            x[i * 2] = 0.0
            x[i * 2 + 1] = 1.0
        x[10 + i] = weights.get(flag, 0.2)
    grades = config.get("grade_thresholds", {})
    x[15] = grades.get("good", 0.85)
    x[16] = grades.get("questionable", 0.65)
    nk = config.get("neurokit", {})
    x[17] = 1.0 if nk.get("enabled", False) else 0.0
    method = nk.get("method", "averageQRS")
    x[18] = float(_NK_METHODS.index(method)) if method in _NK_METHODS else 0.0
    x[19] = nk.get("weight", 0.0)
    return x


def decode_vector(x: np.ndarray) -> Dict[str, Any]:
    """Convert a 20-element vector back into a quality config dict.

    Handles constraint enforcement via min/max swaps so the config
    is always valid (no crashes). The loss function additionally
    penalizes constraint violations to steer the optimizer.
    """
    thresholds = {}
    for i, flag in enumerate(_FLAG_ORDER):
        low = float(x[i * 2])
        high = float(x[i * 2 + 1])
        if flag == "Low_SNR":
            # Inverted: good (high value) must be > bad (low value)
            thresholds[flag] = (max(low, high), min(low, high))
        else:
            # Normal: low must be <= high
            thresholds[flag] = (min(low, high), max(low, high))

    flags_weights = {}
    for i, flag in enumerate(_FLAG_ORDER):
        flags_weights[flag] = float(np.clip(x[10 + i], 0.0, 1.0))

    grade_good = float(np.clip(x[15], 0.3, 1.0))
    grade_quest = float(np.clip(x[16], 0.1, 0.9))
    # Enforce good > questionable
    if grade_good <= grade_quest:
        grade_good, grade_quest = grade_quest + 0.01, grade_good

    nk_enabled = float(x[17]) >= 0.5
    nk_method_idx = int(np.clip(math.floor(x[18]), 0, len(_NK_METHODS) - 1))
    nk_weight = float(np.clip(x[19], 0.0, 1.0))

    return {
        "_preset_name": "optimized",
        "thresholds": thresholds,
        "flags_weights": flags_weights,
        "grade_thresholds": {
            "good": grade_good,
            "questionable": grade_quest,
        },
        "neurokit": {
            "enabled": nk_enabled,
            "method": _NK_METHODS[nk_method_idx],
            "weight": nk_weight,
        },
    }


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective_function(
    x: np.ndarray,
    preprocessed_files: List[PreprocessedFile],
    folder_grades: Dict[str, str],
) -> float:
    """Loss = MSE(actual_score vs grade_target) + penalties.

    Penalties:
    - Classification mismatch: +0.5 per wrong grade
    - Weight sum deviation from 1.0: 0.1 * |sum - 1|
    - Threshold ordering violations: +0.2 each
    """
    config = decode_vector(x)

    # --- Evaluate all files ---
    folder_scores: Dict[str, List[float]] = {}
    for pf in preprocessed_files:
        try:
            result = analyze_holter_quality(
                pf.lead1, pf.lead2, sampling_rate=200, config=config,
            )
            score = result["overall_quality"]
        except Exception:
            score = 0.0
        folder_scores.setdefault(pf.folder, []).append(score)

    # --- MSE loss ---
    mse_sum = 0.0
    n_files = 0
    classification_penalty = 0.0

    for folder, scores in folder_scores.items():
        grade_label = folder_grades.get(folder, "Questionable")
        target = GRADE_TARGETS.get(grade_label, 0.75)
        grade_thresholds = config["grade_thresholds"]

        for score in scores:
            mse_sum += (score - target) ** 2
            n_files += 1

            # Classification penalty
            if score > grade_thresholds["good"]:
                actual_grade = "Good"
            elif score > grade_thresholds["questionable"]:
                actual_grade = "Questionable"
            else:
                actual_grade = "Not usable"
            if actual_grade != grade_label:
                classification_penalty += 0.5

    mse = mse_sum / max(n_files, 1)

    # --- Regularization penalties ---
    weight_sum = sum(config["flags_weights"].values())
    weight_penalty = 0.1 * abs(weight_sum - 1.0)

    # Ordering penalties (raw vector, before decode fixes)
    ordering_penalty = 0.0
    for i, flag in enumerate(_FLAG_ORDER):
        low = float(x[i * 2])
        high = float(x[i * 2 + 1])
        if flag == "Low_SNR":
            if low <= high:
                ordering_penalty += 0.2
        else:
            if low > high:
                ordering_penalty += 0.2
    # Grade ordering
    if float(x[15]) <= float(x[16]):
        ordering_penalty += 0.2

    return mse + classification_penalty + weight_penalty + ordering_penalty


# ---------------------------------------------------------------------------
# Tracker & evaluation helpers
# ---------------------------------------------------------------------------

class ObjectiveTracker:
    """Wraps objective_function to track best loss and eval count."""

    def __init__(
        self,
        preprocessed_files: List[PreprocessedFile],
        folder_grades: Dict[str, str],
    ):
        self.preprocessed_files = preprocessed_files
        self.folder_grades = folder_grades
        self.best_loss = float("inf")
        self.n_evals = 0
        self.history: List[float] = []

    def __call__(self, x: np.ndarray) -> float:
        loss = objective_function(x, self.preprocessed_files, self.folder_grades)
        self.n_evals += 1
        if loss < self.best_loss:
            self.best_loss = loss
        return loss

    def generation_callback(self, xk, convergence):
        """DE callback fired per generation. Records best-so-far."""
        self.history.append(self.best_loss)


def evaluate_config(
    config: Dict[str, Any],
    preprocessed_files: List[PreprocessedFile],
    folder_grades: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Run config against all files, return per-folder results for display."""
    folder_results: Dict[str, List[Dict]] = {}

    for pf in preprocessed_files:
        try:
            result = analyze_holter_quality(
                pf.lead1, pf.lead2, sampling_rate=200, config=config,
            )
            score = result["overall_quality"]
            grade = result["grade"]
        except Exception:
            score = 0.0
            grade = "Not usable"
        folder_results.setdefault(pf.folder, []).append({
            "filename": pf.filename,
            "score": score,
            "grade": grade,
        })

    rows = []
    for folder in sorted(folder_results.keys()):
        files = folder_results[folder]
        avg_score = np.mean([f["score"] for f in files])
        # Majority grade
        grade_counts: Dict[str, int] = {}
        for f in files:
            grade_counts[f["grade"]] = grade_counts.get(f["grade"], 0) + 1
        majority_grade = max(grade_counts, key=grade_counts.get)
        expected = folder_grades.get(folder, "Questionable")
        rows.append({
            "folder": folder,
            "n_files": len(files),
            "expected_grade": expected,
            "avg_score": float(avg_score),
            "actual_grade": majority_grade,
            "match": majority_grade == expected,
        })
    return rows


# ---------------------------------------------------------------------------
# Main optimization runner
# ---------------------------------------------------------------------------

def run_optimization(
    base_folder: str,
    folder_grades: Dict[str, str],
    notch_freqs: List[int],
    current_config: Dict[str, Any],
    thresh_config: Dict[str, Dict],
    maxiter: int = 100,
    popsize: int = 30,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> OptimizationResult:
    """Run differential_evolution to optimize quality config.

    Args:
        base_folder: root data directory
        folder_grades: {folder_name: grade_label}
        notch_freqs: notch filter frequencies
        current_config: current quality config dict (used as seed)
        thresh_config: _THRESH_CONFIG from app.py for bounds
        maxiter: max DE generations
        popsize: DE population size
        seed: random seed for reproducibility
        progress_callback: optional fn(stage: str, progress: float)

    Returns:
        OptimizationResult with best config and comparison data.
    """
    t0 = time.time()

    # 1. Preprocess
    if progress_callback:
        progress_callback("Preprocessing test data...", 0.0)
    preprocessed = preprocess_test_data(base_folder, folder_grades, notch_freqs)
    if not preprocessed:
        raise ValueError("No test files found in selected folders")
    if progress_callback:
        progress_callback("Preprocessing complete", 0.1)

    # 2. Evaluate before
    if progress_callback:
        progress_callback("Evaluating current config...", 0.12)
    before_results = evaluate_config(current_config, preprocessed, folder_grades)

    # 3. Set up optimizer
    bounds = get_param_bounds(thresh_config)
    x0 = encode_config(current_config)
    tracker = ObjectiveTracker(preprocessed, folder_grades)

    # Build initial population seeded with current config
    rng = np.random.default_rng(seed)
    n_params = len(bounds)
    init_pop = np.zeros((popsize, n_params))
    init_pop[0] = x0  # first member = current config
    for i in range(1, popsize):
        for j in range(n_params):
            lo, hi = bounds[j]
            init_pop[i, j] = rng.uniform(lo, hi)

    # DE callback for progress
    generation_count = [0]

    def de_callback(xk, convergence):
        generation_count[0] += 1
        tracker.generation_callback(xk, convergence)
        if progress_callback:
            frac = 0.15 + 0.83 * (generation_count[0] / maxiter)
            progress_callback(
                f"Generation {generation_count[0]}/{maxiter} "
                f"(best loss: {tracker.best_loss:.4f})",
                min(frac, 0.98),
            )

    # 4. Run DE
    if progress_callback:
        progress_callback("Starting optimization...", 0.15)

    result = differential_evolution(
        tracker,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        init=init_pop,
        callback=de_callback,
        tol=1e-6,
        atol=1e-6,
        disp=False,
        polish=True,
    )

    best_config = decode_vector(result.x)

    # 5. Evaluate after
    if progress_callback:
        progress_callback("Evaluating optimized config...", 0.99)
    after_results = evaluate_config(best_config, preprocessed, folder_grades)

    elapsed = time.time() - t0
    if progress_callback:
        progress_callback("Optimization complete!", 1.0)

    return OptimizationResult(
        best_config=best_config,
        best_loss=float(result.fun),
        before_results=before_results,
        after_results=after_results,
        convergence_history=tracker.history,
        n_evaluations=tracker.n_evals,
        elapsed_seconds=elapsed,
    )
