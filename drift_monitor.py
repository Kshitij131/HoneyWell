"""Concept drift monitoring using Population Stability Index (PSI).

Monitors distribution shift in Autoencoder reconstruction errors over time per persona against the baseline training set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import ENTITY_TYPES, ProjectConfig

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    persona: str
    psi: float
    level: str

class DriftMonitor:

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.psi_warning = config.training.drift_psi_warning
        self.psi_alert = config.training.drift_psi_alert
        self.n_bins = config.training.drift_n_bins

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        clip_val = max(
            float(np.percentile(reference, 99.9)),
            float(np.percentile(current, 99.9)),
        )
        
        reference = np.clip(reference, a_min=None, a_max=clip_val)
        current = np.clip(current, a_min=None, a_max=clip_val)

        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.percentile(reference, quantiles)
        
        bins = np.unique(bins)
        
        if len(bins) < 2:
            bins = np.linspace(reference.min(), reference.max(), self.n_bins + 1)
            
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        expected_counts, _ = np.histogram(reference, bins=bins)
        actual_counts, _ = np.histogram(current, bins=bins)

        expected_pct = expected_counts / len(reference)
        actual_pct = actual_counts / len(current)

        eps = 1e-6
        expected_pct = np.clip(expected_pct, eps, None)
        actual_pct = np.clip(actual_pct, eps, None)

        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi = np.sum(psi_values)

        return float(psi)

    def check_drift(
        self,
        current_errors: np.ndarray,
        current_entity_types: np.ndarray,
        reference_errors: np.ndarray,
        reference_entity_types: np.ndarray,
    ) -> Dict[str, DriftResult]:
        results: Dict[str, DriftResult] = {}
        
        for etype in ENTITY_TYPES:
            curr_mask = current_entity_types == etype
            ref_mask = reference_entity_types == etype
            
            curr_vals = current_errors[curr_mask]
            ref_vals = reference_errors[ref_mask]
            
            if len(curr_vals) == 0 or len(ref_vals) == 0:
                results[etype] = DriftResult(etype, 0.0, "normal")
                continue
                
            psi = self.compute_psi(ref_vals, curr_vals)
            
            level = "normal"
            if psi >= self.psi_alert:
                level = "alert"
                logger.error("ALERT: Significant concept drift detected for %s (PSI=%.4f >= %.4f)", etype, psi, self.psi_alert)
            elif psi >= self.psi_warning:
                level = "warning"
                logger.warning("WARNING: Moderate concept drift detected for %s (PSI=%.4f >= %.4f)", etype, psi, self.psi_warning)
                
            results[etype] = DriftResult(etype, psi, level)
            
        return results
