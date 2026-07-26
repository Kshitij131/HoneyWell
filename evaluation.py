"""Comprehensive evaluation suite computing precision, recall, F1, ROC-AUC, PR-AUC, FPR, and latency metrics.

Generates evaluation reports with full attack-wise and persona-wise performance breakdowns.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import ATTACK_TYPES, ENTITY_TYPES, ProjectConfig, get_project_config

logger = logging.getLogger(__name__)

@dataclass
class DistributionStats:

    mean: float
    std: float
    min_val: float
    max_val: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }

@dataclass
class ClassMetrics:

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "support": self.support,
        }

@dataclass
class EvaluationResult:

    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    confusion_mat: List[List[int]]
    false_positive_rate: float
    reconstruction_error_stats: Optional[DistributionStats] = None
    risk_score_stats: Optional[DistributionStats] = None
    benign_risk_stats: Optional[DistributionStats] = None
    attack_risk_stats: Optional[DistributionStats] = None
    attack_wise_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)
    persona_wise_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)
    persona_confusion_matrices: Dict[str, List[List[int]]] = field(default_factory=dict)
    latency_stats: Optional[DistributionStats] = None
    warmup_latency_stats: Optional[DistributionStats] = None
    steady_state_latency_stats: Optional[DistributionStats] = None
    throughput_events_per_sec: Optional[float] = None
    threshold: float = 0.5
    thresholds_dict: Optional[Dict[str, float]] = None
    total_samples: int = 0
    total_positives: int = 0
    total_negatives: int = 0
    roc_curve_data: Optional[Dict[str, List[float]]] = None
    pr_curve_data: Optional[Dict[str, List[float]]] = None
    threshold_metrics: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "pr_auc": round(self.pr_auc, 6),
            "roc_auc": round(self.roc_auc, 6),
            "confusion_matrix": self.confusion_mat,
            "false_positive_rate": round(self.false_positive_rate, 6),
            "threshold": self.threshold,
            "total_samples": self.total_samples,
            "total_positives": self.total_positives,
            "total_negatives": self.total_negatives,
        }
        if self.thresholds_dict:
            result["thresholds_dict"] = self.thresholds_dict
        if self.reconstruction_error_stats is not None:
            result["reconstruction_error_distribution"] = self.reconstruction_error_stats.to_dict()
        if self.risk_score_stats is not None:
            result["risk_score_distribution"] = self.risk_score_stats.to_dict()
        if self.benign_risk_stats is not None:
            result["benign_risk_distribution"] = self.benign_risk_stats.to_dict()
        if self.attack_risk_stats is not None:
            result["attack_risk_distribution"] = self.attack_risk_stats.to_dict()
        if self.attack_wise_metrics:
            result["attack_wise_metrics"] = {k: v.to_dict() for k, v in self.attack_wise_metrics.items()}
        if self.persona_wise_metrics:
            result["persona_wise_metrics"] = {k: v.to_dict() for k, v in self.persona_wise_metrics.items()}
        if self.persona_confusion_matrices:
            result["persona_confusion_matrices"] = self.persona_confusion_matrices
        if self.latency_stats is not None:
            result["latency_stats_ms"] = self.latency_stats.to_dict()
        if self.warmup_latency_stats is not None:
            result["warmup_latency_stats_ms"] = self.warmup_latency_stats.to_dict()
        if self.steady_state_latency_stats is not None:
            result["steady_state_latency_stats_ms"] = self.steady_state_latency_stats.to_dict()
        if self.throughput_events_per_sec is not None:
            result["throughput_events_per_sec"] = round(self.throughput_events_per_sec, 2)
        if self.roc_curve_data is not None:
            result["roc_curve_data"] = self.roc_curve_data
        if self.pr_curve_data is not None:
            result["pr_curve_data"] = self.pr_curve_data
        if self.threshold_metrics is not None:
            result["threshold_metrics"] = self.threshold_metrics
        return result

def compute_distribution_stats(values: np.ndarray) -> DistributionStats:
    if len(values) == 0:
        return DistributionStats(
            mean=0.0, std=0.0, min_val=0.0, max_val=0.0,
            p25=0.0, p50=0.0, p75=0.0, p90=0.0, p95=0.0, p99=0.0,
        )
    return DistributionStats(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        p25=float(np.percentile(values, 25)),
        p50=float(np.percentile(values, 50)),
        p75=float(np.percentile(values, 75)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )

def _downsample_curve(x: np.ndarray, y: np.ndarray, t: np.ndarray, n_points: int = 100) -> Tuple[List[float], List[float], List[float]]:
    if len(x) <= n_points:
        return x.tolist(), y.tolist(), t.tolist()
    indices = np.linspace(0, len(x) - 1, n_points, dtype=int)
    return x[indices].tolist(), y[indices].tolist(), t[indices].tolist()

def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    if (fp + tn) == 0:
        return 0.0
    return float(fp / (fp + tn))

def evaluate_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Union[float, Dict[str, float]] = 0.5,
    reconstruction_errors: Optional[np.ndarray] = None,
    risk_scores: Optional[np.ndarray] = None,
    attack_types: Optional[np.ndarray] = None,
    entity_types: Optional[np.ndarray] = None,
    latencies_ms: Optional[np.ndarray] = None,
    warmup_calls: int = 3,
    total_inference_time_sec: Optional[float] = None,
    total_samples_inferred: Optional[int] = None,
    predictions: Optional[np.ndarray] = None,
) -> EvaluationResult:
    
    thresholds_dict = None
    if predictions is not None:
        y_pred = np.asarray(predictions, dtype=int)
        threshold_val = 0.0
        thresholds_dict = threshold if isinstance(threshold, dict) else None
    elif isinstance(threshold, dict) and entity_types is not None:
        y_pred = np.zeros_like(y_scores, dtype=int)
        for i, etype in enumerate(entity_types):
            t = threshold.get(etype, 0.5)
            if y_scores[i] >= t:
                y_pred[i] = 1
        threshold_val = sum(threshold.values()) / len(threshold) if threshold else 0.5
        thresholds_dict = threshold
    else:
        t = threshold if isinstance(threshold, float) else 0.5
        y_pred = (y_scores >= t).astype(int)
        threshold_val = t

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_val = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc_val = float(roc_auc_score(y_true, y_scores))
        fpr_c, tpr_c, thresh_roc = roc_curve(y_true, y_scores)
        fpr_ds, tpr_ds, thresh_roc_ds = _downsample_curve(fpr_c, tpr_c, thresh_roc)
        roc_curve_data = {"fpr": fpr_ds, "tpr": tpr_ds, "thresholds": thresh_roc_ds}
    except ValueError:
        roc_auc_val = 0.0
        roc_curve_data = None
        logger.warning("ROC-AUC could not be computed (single class present).")

    try:
        pr_auc_val = float(average_precision_score(y_true, y_scores))
        p_c, r_c, thresh_pr = precision_recall_curve(y_true, y_scores)
        thresh_pr = np.append(thresh_pr, 1.0)
        p_ds, r_ds, thresh_pr_ds = _downsample_curve(p_c, r_c, thresh_pr)
        pr_curve_data = {"precision": p_ds, "recall": r_ds, "thresholds": thresh_pr_ds}
        
        threshold_metrics = {
            "thresholds": thresh_pr_ds,
            "precision": p_ds,
            "recall": r_ds,
            "f1": [2 * p * r / (p + r) if (p+r) > 0 else 0.0 for p, r in zip(p_ds, r_ds)]
        }
    except ValueError:
        pr_auc_val = 0.0
        pr_curve_data = None
        threshold_metrics = None
        logger.warning("PR-AUC could not be computed.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_list = cm.tolist()
    fpr = compute_fpr(y_true, y_pred)

    recon_stats = compute_distribution_stats(reconstruction_errors) if reconstruction_errors is not None else None
    risk_stats = compute_distribution_stats(risk_scores) if risk_scores is not None else None
    
    benign_risk_stats = None
    attack_risk_stats = None
    if risk_scores is not None:
        benign_risk_stats = compute_distribution_stats(risk_scores[y_true == 0])
        attack_risk_stats = compute_distribution_stats(risk_scores[y_true == 1])

    lat_stats = compute_distribution_stats(latencies_ms) if latencies_ms is not None else None
    warmup_lat_stats = None
    steady_lat_stats = None
    throughput = None
    if latencies_ms is not None and len(latencies_ms) > warmup_calls:
        warmup_lat_stats = compute_distribution_stats(latencies_ms[:warmup_calls])
        steady_lat_stats = compute_distribution_stats(latencies_ms[warmup_calls:])
        logger.info(
            "Latency split — warm-up P50=%.2fms (n=%d), steady-state P50=%.2fms P95=%.2fms P99=%.2fms (n=%d)",
            warmup_lat_stats.p50, warmup_calls,
            steady_lat_stats.p50, steady_lat_stats.p95, steady_lat_stats.p99,
            len(latencies_ms) - warmup_calls,
        )
    if total_inference_time_sec is not None and total_samples_inferred is not None and total_inference_time_sec > 0:
        throughput = total_samples_inferred / total_inference_time_sec
        logger.info("Throughput: %.2f events/sec", throughput)

    atk_metrics: Dict[str, ClassMetrics] = {}
    if attack_types is not None:
        for atype in ATTACK_TYPES:
            mask = attack_types == atype
            if mask.sum() == 0:
                continue
            atk_metrics[atype] = ClassMetrics(
                precision=0.0,
                recall=float(y_pred[mask].mean()),
                f1=0.0,
                support=int(mask.sum()),
            )

    persona_metrics: Dict[str, ClassMetrics] = {}
    persona_cms: Dict[str, List[List[int]]] = {}
    if entity_types is not None:
        for etype in ENTITY_TYPES:
            mask = entity_types == etype
            if mask.sum() == 0:
                continue
            p_true = y_true[mask]
            p_pred = y_pred[mask]
            persona_metrics[etype] = ClassMetrics(
                precision=float(precision_score(p_true, p_pred, zero_division=0)),
                recall=float(recall_score(p_true, p_pred, zero_division=0)),
                f1=float(f1_score(p_true, p_pred, zero_division=0)),
                support=int(mask.sum()),
            )
            persona_cms[etype] = confusion_matrix(p_true, p_pred, labels=[0, 1]).tolist()

    total_pos = int(y_true.sum())
    total_neg = int(len(y_true) - total_pos)

    result = EvaluationResult(
        precision=prec,
        recall=rec,
        f1=f1_val,
        pr_auc=pr_auc_val,
        roc_auc=roc_auc_val,
        confusion_mat=cm_list,
        false_positive_rate=fpr,
        reconstruction_error_stats=recon_stats,
        risk_score_stats=risk_stats,
        benign_risk_stats=benign_risk_stats,
        attack_risk_stats=attack_risk_stats,
        attack_wise_metrics=atk_metrics,
        persona_wise_metrics=persona_metrics,
        persona_confusion_matrices=persona_cms,
        latency_stats=lat_stats,
        warmup_latency_stats=warmup_lat_stats,
        steady_state_latency_stats=steady_lat_stats,
        throughput_events_per_sec=throughput,
        threshold=threshold_val,
        thresholds_dict=thresholds_dict,
        total_samples=len(y_true),
        total_positives=total_pos,
        total_negatives=total_neg,
        roc_curve_data=roc_curve_data,
        pr_curve_data=pr_curve_data,
        threshold_metrics=threshold_metrics,
    )

    logger.info(
        "Evaluation complete — P=%.4f, R=%.4f, F1=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f, FPR=%.4f",
        prec, rec, f1_val, roc_auc_val, pr_auc_val, fpr,
    )

    return result

def generate_report(
    result: EvaluationResult,
    output_dir: Path,
    report_name: str = "evaluation_report",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dict = result.to_dict()

    json_path = output_dir / f"{report_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    txt_path = output_dir / f"{report_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BEHAVIORAL ANOMALY DETECTION — EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Precision     : {result.precision:.6f}\n")
        f.write(f"  Recall        : {result.recall:.6f}\n")
        f.write(f"  F1-Score      : {result.f1:.6f}\n")
        f.write(f"  ROC-AUC       : {result.roc_auc:.6f}\n")
        f.write(f"  PR-AUC        : {result.pr_auc:.6f}\n")
        f.write(f"  FPR           : {result.false_positive_rate:.6f}\n")
        
        if result.thresholds_dict:
            f.write(f"  Thresholds    : {json.dumps(result.thresholds_dict)}\n")
        else:
            f.write(f"  Threshold     : {result.threshold:.4f}\n")
            
        f.write(f"  Total Samples : {result.total_samples}\n")
        f.write(f"  Positives     : {result.total_positives}\n")
        f.write(f"  Negatives     : {result.total_negatives}\n\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        cm = result.confusion_mat
        f.write(f"  TN={cm[0][0]:>7}  FP={cm[0][1]:>7}\n")
        f.write(f"  FN={cm[1][0]:>7}  TP={cm[1][1]:>7}\n\n")

        if result.risk_score_stats is not None:
            f.write("HYBRID RISK SCORE DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            stats = result.risk_score_stats
            f.write(f"  Mean : {stats.mean:.6f}  Std : {stats.std:.6f}\n")
            f.write(f"  Min  : {stats.min_val:.6f}  Max : {stats.max_val:.6f}\n")
            f.write(f"  P50  : {stats.p50:.6f}  P95 : {stats.p95:.6f}\n\n")

        if result.steady_state_latency_stats is not None:
            f.write("LATENCY (STEADY-STATE, EXCL. WARM-UP)\n")
            f.write("-" * 40 + "\n")
            ss = result.steady_state_latency_stats
            f.write(f"  P50  : {ss.p50:.2f}ms  P95 : {ss.p95:.2f}ms  P99 : {ss.p99:.2f}ms\n")
            if result.warmup_latency_stats is not None:
                wu = result.warmup_latency_stats
                f.write(f"  Warm-up P50: {wu.p50:.2f}ms (excluded from above)\n")
            if result.throughput_events_per_sec is not None:
                f.write(f"  Throughput  : {result.throughput_events_per_sec:.2f} events/sec\n")
            f.write("\n")

        if result.attack_wise_metrics:
            f.write("ATTACK-WISE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"  {'Attack Type':<30} {'Recall':>8} {'N':>6}\n")
            for atype, m in sorted(result.attack_wise_metrics.items()):
                f.write(
                    f"  {atype:<30} {m.recall:>8.4f} {m.support:>6}\n"
                )
            f.write("\n")

        if result.persona_wise_metrics:
            f.write("PERSONA-WISE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"  {'Persona':<30} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}\n")
            for etype, m in sorted(result.persona_wise_metrics.items()):
                f.write(
                    f"  {etype:<30} {m.precision:>8.4f} {m.recall:>8.4f} "
                    f"{m.f1:>8.4f} {m.support:>6}\n"
                )
            f.write("\n")

        f.write("=" * 80 + "\n")

    logger.info("Reports saved to %s (.json and .txt)", output_dir)
    return json_path

class LatencyTracker:
    def __init__(self) -> None:
        self._latencies: List[float] = []
        self._start_time: float = 0.0

    def __enter__(self) -> "LatencyTracker":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._latencies.append(elapsed_ms)

    def record(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def get_latencies(self) -> np.ndarray:
        return np.array(self._latencies, dtype=np.float64)

    def get_stats(self) -> DistributionStats:
        return compute_distribution_stats(self.get_latencies())

def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
    constraint: float = 0.90,
    thresholds: Optional[np.ndarray] = None,
    attack_labels: Optional[np.ndarray] = None,
    min_attack_recall: float = 0.0,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
        
    best_threshold = 0.5
    best_value = -2.0
    
    if len(y_true) == 0:
        return best_threshold, best_value

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        
        if metric == "f1":
            val = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "balanced_accuracy":
            val = float(balanced_accuracy_score(y_true, y_pred))
        elif metric == "precision_at_recall":
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            if rec >= constraint:
                val = float(precision_score(y_true, y_pred, zero_division=0))
            else:
                val = 0.0
        elif metric == "recall_at_fpr":
            fpr = compute_fpr(y_true, y_pred)
            if fpr <= constraint:
                val = float(recall_score(y_true, y_pred, zero_division=0))
            else:
                val = 0.0
        else:
            val = float(f1_score(y_true, y_pred, zero_division=0))
            
        if min_attack_recall > 0.0 and attack_labels is not None:
            min_rec = 1.0
            for atype in np.unique(attack_labels):
                if atype == "benign":
                    continue
                mask_atype = (attack_labels == atype)
                if mask_atype.sum() > 0:
                    rec = (y_pred[mask_atype] == 1).sum() / mask_atype.sum()
                    min_rec = min(min_rec, float(rec))
            if min_rec < min_attack_recall:
                val = -1.0
            
        if val > best_value:
            best_value = val
            best_threshold = float(t)
            
    if best_value == -1.0:
        best_threshold = float(thresholds[0])
        best_value = 0.0
            
    return best_threshold, best_value
