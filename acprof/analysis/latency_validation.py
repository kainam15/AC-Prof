"""Cross-validation and quality gates for latency models."""

import math
from typing import Any

from acprof.analysis.latency_model import (
    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE,
    LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE,
    LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR,
    LATENCY_MODEL_MAX_VALIDATION_MAPE,
    LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE,
    LATENCY_MODEL_MIN_VALIDATION_POINTS,
    LATENCY_MODEL_MIN_VALIDATION_R2,
    LATENCY_MODEL_RESOURCE_FEATURES,
    fit_log_latency_model,
    point_key,
    predict_latency,
    regression_metrics,
)


def _prediction_summary(
    rows: list[Any],
    predictions: dict[tuple[str, float, float, float], float],
) -> dict[str, Any]:
    predicted_rows = [row for row in rows if point_key(row) in predictions]
    actuals = [float(row.latency_s) for row in predicted_rows]
    predicted = [predictions[point_key(row)] for row in predicted_rows]
    case_relative_errors = [
        abs(actual - prediction) / actual
        for actual, prediction in zip(actuals, predicted, strict=True)
    ]
    worst_case_idx = (
        None
        if not predicted_rows
        else max(range(len(predicted_rows)), key=lambda idx: case_relative_errors[idx])
    )
    worst_case = None if worst_case_idx is None else predicted_rows[worst_case_idx]
    return {
        "test_case_rows": len(predicted_rows),
        "worst_case_relative_error": (
            None if worst_case_idx is None else case_relative_errors[worst_case_idx]
        ),
        "worst_case_configuration": (
            None
            if worst_case is None
            else {
                "cpu_cores": float(worst_case.cpu_cores),
                "mem_cap_gb": float(worst_case.mem_cap_gb),
                "input_scale": float(worst_case.input_scale),
            }
        ),
        "metrics": regression_metrics(actuals, predicted),
    }


def resource_configuration_validation(
    rows: list[Any],
    hardware_model: str,
) -> tuple[dict[str, Any], dict[tuple[str, float, float, float], float]]:
    """Leave one complete (CPU, memory) configuration out per fold."""
    configurations = sorted({
        (float(row.cpu_cores), float(row.mem_cap_gb)) for row in rows
    })
    if len(configurations) < 3:
        return {
            "available": False,
            "reason": (
                "need at least 3 resource configurations so each held-out fold "
                "retains at least 2 training configurations"
            ),
            "folds": 0,
            "completed_folds": 0,
            "failed_folds": [],
            "r2_quality_gate_applicable": True,
            "train_test_group_overlap_count": 0,
            "metrics": regression_metrics([], []),
        }, {}

    predictions: dict[tuple[str, float, float, float], float] = {}
    failed_folds = []
    fold_details = []
    train_test_group_overlap_count = 0
    completed_folds = 0
    for cpu_cores, mem_cap_gb in configurations:
        test_rows = [
            row for row in rows
            if float(row.cpu_cores) == cpu_cores and float(row.mem_cap_gb) == mem_cap_gb
        ]
        train_rows = [
            row for row in rows
            if not (float(row.cpu_cores) == cpu_cores and float(row.mem_cap_gb) == mem_cap_gb)
        ]
        train_groups = {(float(row.cpu_cores), float(row.mem_cap_gb)) for row in train_rows}
        test_groups = {(float(row.cpu_cores), float(row.mem_cap_gb)) for row in test_rows}
        fold_overlap_count = len(train_groups & test_groups)
        train_test_group_overlap_count += fold_overlap_count
        fold_detail: dict[str, Any] = {
            "held_out_cpu_cores": cpu_cores,
            "held_out_mem_cap_gb": mem_cap_gb,
            "train_case_rows": len(train_rows),
            "test_case_rows": len(test_rows),
            "train_test_group_overlap_count": fold_overlap_count,
        }
        try:
            calibrate_tail = hardware_model != "gpu" or (
                max(float(row.input_scale) for row in test_rows)
                > max(float(row.input_scale) for row in train_rows)
            )
            fold_model = fit_log_latency_model(
                train_rows, hardware_model, calibrate_gpu_upper_tail=calibrate_tail,
            )
            if not (
                LATENCY_MODEL_RESOURCE_FEATURES[hardware_model]
                & set(fold_model["selected_feature_names"])
            ):
                raise ValueError("held-out fold has no identifiable CPU or memory predictor")
        except ValueError as exc:
            fold_detail["status"] = "failed"
            fold_detail["reason"] = str(exc)
            fold_details.append(fold_detail)
            failed_folds.append({
                "cpu_cores": cpu_cores,
                "mem_cap_gb": mem_cap_gb,
                "reason": str(exc),
            })
            continue

        fold_predictions = [predict_latency(row, fold_model) for row in test_rows]
        for row, prediction in zip(test_rows, fold_predictions, strict=True):
            predictions[point_key(row)] = prediction
        fold_detail["status"] = "ok"
        fold_detail["metrics"] = regression_metrics(
            [float(row.latency_s) for row in test_rows], fold_predictions,
        )
        fold_details.append(fold_detail)
        completed_folds += 1

    summary = _prediction_summary(rows, predictions)
    completed_fold_details = [fold for fold in fold_details if fold.get("status") == "ok"]
    worst_relative_mae_fold = (
        None
        if not completed_fold_details
        else max(
            completed_fold_details,
            key=lambda fold: (
                math.inf
                if fold["metrics"].get("relative_mae") is None
                else float(fold["metrics"]["relative_mae"])
            ),
        )
    )
    return {
        "available": not failed_folds and summary["test_case_rows"] == len(rows),
        "method": "leave-one-(cpu_cores,mem_cap_gb)-configuration-out",
        "r2_quality_gate_applicable": True,
        "group_columns": ["cpu_cores", "mem_cap_gb"],
        "all_input_scales_move_with_held_out_configuration": True,
        "folds": len(configurations),
        "completed_folds": completed_folds,
        "failed_folds": failed_folds,
        "fold_details": fold_details,
        "train_test_group_overlap_count": train_test_group_overlap_count,
        "test_case_rows": summary["test_case_rows"],
        "worst_fold_relative_mae": (
            None if worst_relative_mae_fold is None
            else worst_relative_mae_fold["metrics"]["relative_mae"]
        ),
        "worst_fold_configuration": (
            None
            if worst_relative_mae_fold is None
            else {
                "cpu_cores": worst_relative_mae_fold["held_out_cpu_cores"],
                "mem_cap_gb": worst_relative_mae_fold["held_out_mem_cap_gb"],
            }
        ),
        "worst_case_relative_error": summary["worst_case_relative_error"],
        "worst_case_configuration": summary["worst_case_configuration"],
        "metrics": summary["metrics"],
    }, predictions


def input_scale_validation(
    rows: list[Any],
    hardware_model: str,
) -> tuple[dict[str, Any], dict[tuple[str, float, float, float], float]]:
    """Train on smaller scales and hold out the largest scale for extrapolation."""
    input_scales = sorted({float(row.input_scale) for row in rows})
    if len(input_scales) < 3:
        return {
            "available": False,
            "reason": (
                "need at least 3 input scales so the maximum can be held out "
                "while at least 2 distinct scales remain for training"
            ),
            "held_out_input_scale": None,
            "group_columns": ["input_scale"],
            "r2_quality_gate_applicable": False,
            "r2_quality_gate_note": (
                "A single held-out scale tests level extrapolation; R-squared "
                "within that scale only measures resource-configuration "
                "variation and is not used as an extrapolation gate."
            ),
            "train_test_group_overlap_count": 0,
            "metrics": regression_metrics([], []),
        }, {}

    held_out_scale = input_scales[-1]
    train_rows = [row for row in rows if float(row.input_scale) < held_out_scale]
    test_rows = [row for row in rows if float(row.input_scale) == held_out_scale]
    train_scale_groups = {float(row.input_scale) for row in train_rows}
    test_scale_groups = {float(row.input_scale) for row in test_rows}
    train_test_group_overlap_count = len(train_scale_groups & test_scale_groups)
    try:
        scale_model = fit_log_latency_model(train_rows, hardware_model)
        if "log_input_scale" not in scale_model["selected_feature_names"]:
            raise ValueError("maximum-scale training fold has no identifiable input-scale predictor")
    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "held_out_input_scale": held_out_scale,
            "group_columns": ["input_scale"],
            "r2_quality_gate_applicable": False,
            "r2_quality_gate_note": (
                "A single held-out scale tests level extrapolation; R-squared "
                "within that scale only measures resource-configuration variation "
                "and is reported for context, not used as an extrapolation gate."
            ),
            "train_input_scale_max": max(
                (float(row.input_scale) for row in train_rows), default=None,
            ),
            "train_test_group_overlap_count": train_test_group_overlap_count,
            "metrics": regression_metrics([], []),
        }, {}

    predictions = {point_key(row): predict_latency(row, scale_model) for row in test_rows}
    summary = _prediction_summary(rows, predictions)
    return {
        "available": True,
        "method": "forward holdout of maximum input_scale",
        "group_columns": ["input_scale"],
        "r2_quality_gate_applicable": False,
        "r2_quality_gate_note": (
            "A single held-out scale tests level extrapolation; R-squared "
            "within that scale only measures resource-configuration variation "
            "and is reported for context, not used as an extrapolation gate."
        ),
        "held_out_input_scale": held_out_scale,
        "train_input_scale_max": max(float(row.input_scale) for row in train_rows),
        "test_input_scale_min": min(float(row.input_scale) for row in test_rows),
        "strict_extrapolation": (
            max(float(row.input_scale) for row in train_rows)
            < min(float(row.input_scale) for row in test_rows)
        ),
        "train_test_group_overlap_count": train_test_group_overlap_count,
        "train_case_rows": len(train_rows),
        "test_case_rows": summary["test_case_rows"],
        "worst_case_relative_error": summary["worst_case_relative_error"],
        "worst_case_configuration": summary["worst_case_configuration"],
        "metrics": summary["metrics"],
    }, predictions


def validation_quality_failures(validation_name: str, validation: dict[str, Any]) -> list[str]:
    if not validation.get("available"):
        return [
            f"{validation_name} unavailable: "
            f"{validation.get('reason', 'incomplete validation folds')}"
        ]

    metrics = validation["metrics"]
    failures = []
    if int(metrics.get("prediction_count") or 0) < LATENCY_MODEL_MIN_VALIDATION_POINTS:
        failures.append(
            f"{validation_name} has fewer than {LATENCY_MODEL_MIN_VALIDATION_POINTS} predictions"
        )
    if int(validation.get("train_test_group_overlap_count") or 0) > 0:
        failures.append(f"{validation_name} has train/test group leakage")
    failed_quality_folds = []
    for fold_detail in validation.get("fold_details", []):
        if fold_detail.get("status") != "ok":
            continue
        fold_metrics = fold_detail["metrics"]
        fold_relative_mae = fold_metrics.get("relative_mae")
        fold_mape = fold_metrics.get("mean_absolute_percentage_error")
        if (
            fold_relative_mae is None
            or not math.isfinite(float(fold_relative_mae))
            or float(fold_relative_mae) > LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE
            or fold_mape is None
            or not math.isfinite(float(fold_mape))
            or float(fold_mape) > LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE
            or int(fold_metrics.get("nonfinite_prediction_count") or 0) > 0
            or int(fold_metrics.get("nonpositive_prediction_count") or 0) > 0
        ):
            failed_quality_folds.append(fold_detail)
    if failed_quality_folds:
        worst_fold = max(
            failed_quality_folds,
            key=lambda fold_detail: max(
                math.inf
                if (
                    fold_detail["metrics"].get("relative_mae") is None
                    or not math.isfinite(float(fold_detail["metrics"]["relative_mae"]))
                )
                else float(fold_detail["metrics"]["relative_mae"])
                / LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE,
                math.inf
                if (
                    fold_detail["metrics"].get("mean_absolute_percentage_error") is None
                    or not math.isfinite(
                        float(fold_detail["metrics"]["mean_absolute_percentage_error"])
                    )
                )
                else float(fold_detail["metrics"]["mean_absolute_percentage_error"])
                / LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE,
            ),
        )
        failures.append(
            f"{validation_name} has {len(failed_quality_folds)} held-out "
            "configuration fold(s) above the relative-MAE/MAPE/validity threshold; "
            f"worst=(cpu_cores={worst_fold['held_out_cpu_cores']},"
            f"mem_cap_gb={worst_fold['held_out_mem_cap_gb']}), "
            f"relative_mae={worst_fold['metrics'].get('relative_mae')!r}, "
            "mean_absolute_percentage_error="
            f"{worst_fold['metrics'].get('mean_absolute_percentage_error')!r}"
        )
    worst_case_relative_error = validation.get("worst_case_relative_error")
    if worst_case_relative_error is not None and (
        not math.isfinite(float(worst_case_relative_error))
        or float(worst_case_relative_error) > LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR
    ):
        failures.append(
            f"{validation_name} worst case relative_error="
            f"{worst_case_relative_error!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR}; "
            f"configuration={validation.get('worst_case_configuration')!r}"
        )
    if validation.get("r2_quality_gate_applicable", True):
        r2 = metrics.get("r2")
        if r2 is None or not math.isfinite(float(r2)) or float(r2) < LATENCY_MODEL_MIN_VALIDATION_R2:
            failures.append(f"{validation_name} r2={r2!r} is below {LATENCY_MODEL_MIN_VALIDATION_R2}")
    relative_mae = metrics.get("relative_mae")
    if (
        relative_mae is None
        or not math.isfinite(float(relative_mae))
        or float(relative_mae) > LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE
    ):
        failures.append(
            f"{validation_name} relative_mae={relative_mae!r} exceeds "
            f"{LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE}"
        )
    mean_absolute_percentage_error = metrics.get("mean_absolute_percentage_error")
    if (
        mean_absolute_percentage_error is None
        or not math.isfinite(float(mean_absolute_percentage_error))
        or float(mean_absolute_percentage_error) > LATENCY_MODEL_MAX_VALIDATION_MAPE
    ):
        failures.append(
            f"{validation_name} mean_absolute_percentage_error="
            f"{mean_absolute_percentage_error!r} exceeds {LATENCY_MODEL_MAX_VALIDATION_MAPE}"
        )
    if int(metrics.get("nonfinite_prediction_count") or 0) > 0:
        failures.append(f"{validation_name} contains non-finite predictions")
    if int(metrics.get("nonpositive_prediction_count") or 0) > 0:
        failures.append(f"{validation_name} contains non-positive predictions")
    return failures


def validation_is_evaluable(validation: dict[str, Any]) -> bool:
    if not validation.get("available"):
        return False
    metrics = validation.get("metrics", {})
    metric_names = ["relative_mae", "mean_absolute_percentage_error"]
    if validation.get("r2_quality_gate_applicable", True):
        metric_names.append("r2")
    return bool(
        int(metrics.get("prediction_count") or 0) >= LATENCY_MODEL_MIN_VALIDATION_POINTS
        and all(
            metrics.get(metric_name) is not None
            and math.isfinite(float(metrics[metric_name]))
            for metric_name in metric_names
        )
    )
