"""数值延迟模型拟合、预测及验证；不依赖绘图层。"""

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from acprof.gpu_mode import normalize_gpu_mode

LATENCY_MODEL_FEATURES = {
    "cpu": [
        "intercept",
        "log_input_scale",
        "log_input_scale_squared",
        "log_cpu_cores",
        "log_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_log_cpu_cores",
        "log_input_scale_x_log_mem_cap_gb",
        "log_cpu_cores_x_log_mem_cap_gb",
    ],
    "gpu": [
        "intercept",
        "log_input_scale",
        "inverse_cpu_cores",
        "inverse_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_inverse_cpu_cores",
        "log_input_scale_x_inverse_cpu_cores_squared",
    ],
}
LATENCY_MODEL_RESOURCE_FEATURES = {
    "cpu": {
        "log_cpu_cores",
        "log_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_log_cpu_cores",
        "log_input_scale_x_log_mem_cap_gb",
        "log_cpu_cores_x_log_mem_cap_gb",
    },
    "gpu": {
        "inverse_cpu_cores",
        "inverse_cpu_cores_squared",
        "log_mem_cap_gb",
        "log_input_scale_x_inverse_cpu_cores",
        "log_input_scale_x_inverse_cpu_cores_squared",
    },
}
LATENCY_MODEL_FORMULAS = {
    "cpu": (
        "latency_s = exp(intercept + log_input_scale "
        "+ log_input_scale_squared "
        "+ log_cpu_cores + log_cpu_cores_squared + log_mem_cap_gb "
        "+ log_input_scale_x_log_cpu_cores "
        "+ log_input_scale_x_log_mem_cap_gb "
        "+ log_cpu_cores_x_log_mem_cap_gb)"
    ),
    "gpu": (
        "within-range latency_s = exp(intercept + log_input_scale "
        "+ piecewise_linear_log_input_scale_hinges "
        "+ inverse_cpu_cores + inverse_cpu_cores_squared "
        "+ log_mem_cap_gb + log_input_scale_x_inverse_cpu_cores "
        "+ log_input_scale_x_inverse_cpu_cores_squared); when activated, "
        "upper-tail latency_s = boundary_spline_latency_s + "
        "affine_tail(input_scale) - affine_tail(boundary_input_scale)"
    ),
}
LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES = [
    "intercept",
    "input_scale",
    "inverse_cpu_cores",
    "inverse_cpu_cores_squared",
    "log_mem_cap_gb",
    "input_scale_x_inverse_cpu_cores",
    "input_scale_x_inverse_cpu_cores_squared",
]
# A GPU log-spline is excellent inside the profiled scale range, but extending
# one noisy boundary segment can be brittle.  When a nested one-step-forward
# check exceeds this MAPE, upper extrapolation switches to a continuous affine
# work tail fitted in latency space.
LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE = 0.05
LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO = 10.0
LATENCY_MODEL_MIN_VALIDATION_R2 = 0.80
LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE = 0.20
LATENCY_MODEL_MAX_VALIDATION_MAPE = 0.20
LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE = 0.30
LATENCY_MODEL_MAX_CONFIGURATION_FOLD_MAPE = 0.30
LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR = 0.30
# Compatibility alias retained for callers that used the original max-scale-only
# threshold name.
LATENCY_MODEL_MIN_VALIDATION_POINTS = 4


def _hardware_model_name(gpu_mode: object) -> str:
    return "gpu" if normalize_gpu_mode(gpu_mode) == "on" else "cpu"


def _format_input_scale_knot(input_scale: float) -> str:
    return (
        format(float(input_scale), ".12g")
        .replace("-", "neg_")
        .replace(".", "p")
    )


def model_feature_names(
    hardware_model: str,
    input_scale_spline_knots: list[float] | None = None,
) -> list[str]:
    if hardware_model == "cpu":
        return list(LATENCY_MODEL_FEATURES[hardware_model])
    if hardware_model == "gpu":
        knots = input_scale_spline_knots or []
        return [
            "intercept",
            "log_input_scale",
            *[
                f"log_input_scale_hinge_at_{_format_input_scale_knot(knot)}"
                for knot in knots
            ],
            *LATENCY_MODEL_FEATURES[hardware_model][2:],
        ]
    raise ValueError(f"unknown latency hardware model: {hardware_model}")


def _gpu_input_scale_spline_knots(rows: list) -> list[float]:
    """Use every interior observed scale as a shared log-scale spline knot."""
    input_scales = sorted({float(row.input_scale) for row in rows})
    return input_scales[1:-1]


def _model_features(
    row,
    hardware_model: str,
    input_scale_spline_knots: list[float] | None = None,
) -> list[float]:
    input_scale = float(row.input_scale)
    cpu_cores = float(row.cpu_cores)
    mem_cap_gb = float(row.mem_cap_gb)
    log_input_scale = math.log(input_scale)
    log_mem_cap_gb = math.log(mem_cap_gb)

    if hardware_model == "cpu":
        log_cpu_cores = math.log(cpu_cores)
        return [
            1.0,
            log_input_scale,
            log_input_scale**2,
            log_cpu_cores,
            log_cpu_cores**2,
            log_mem_cap_gb,
            log_input_scale * log_cpu_cores,
            log_input_scale * log_mem_cap_gb,
            log_cpu_cores * log_mem_cap_gb,
        ]
    if hardware_model == "gpu":
        inverse_cpu_cores = 1.0 / cpu_cores
        inverse_cpu_cores_squared = inverse_cpu_cores**2
        spline_knots = input_scale_spline_knots or []
        return [
            1.0,
            log_input_scale,
            *[
                max(0.0, log_input_scale - math.log(float(knot)))
                for knot in spline_knots
            ],
            inverse_cpu_cores,
            inverse_cpu_cores_squared,
            log_mem_cap_gb,
            log_input_scale * inverse_cpu_cores,
            log_input_scale * inverse_cpu_cores_squared,
        ]
    raise ValueError(f"unknown latency hardware model: {hardware_model}")


def _fit_rank_revealing_linear_model(
    feature_matrix: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Solve a standardized linear model after removing dependent columns."""
    if feature_matrix.ndim != 2 or target.ndim != 1:
        raise ValueError("model matrix and target have invalid dimensions")
    if feature_matrix.shape[0] != target.shape[0]:
        raise ValueError("model matrix and target row counts differ")
    if feature_matrix.shape[1] < 2:
        raise ValueError("model matrix needs an intercept and a predictor")
    if not np.isfinite(feature_matrix).all() or not np.isfinite(target).all():
        raise ValueError("model inputs contain non-finite values")

    means = np.zeros(feature_matrix.shape[1], dtype=float)
    scales = np.ones(feature_matrix.shape[1], dtype=float)
    standardized = np.zeros_like(feature_matrix)
    standardized[:, 0] = 1.0
    for feature_idx in range(1, feature_matrix.shape[1]):
        column = feature_matrix[:, feature_idx]
        means[feature_idx] = float(column.mean())
        scales[feature_idx] = float(column.std())
        if scales[feature_idx] > 1e-12:
            standardized[:, feature_idx] = (
                column - means[feature_idx]
            ) / scales[feature_idx]

    selected_indices: list[int] = []
    for feature_idx in range(standardized.shape[1]):
        candidate_indices = selected_indices + [feature_idx]
        candidate = standardized[:, candidate_indices]
        if int(np.linalg.matrix_rank(candidate)) > len(selected_indices):
            selected_indices.append(feature_idx)

    if len(selected_indices) < 2:
        raise ValueError("model matrix has no usable varying predictor")

    selected_matrix = standardized[:, selected_indices]
    standardized_coefficients, _, rank, singular_values = np.linalg.lstsq(
        selected_matrix,
        target,
        rcond=None,
    )
    if int(rank) != len(selected_indices):
        raise ValueError(
            f"rank-deficient model matrix: rank={int(rank)}, "
            f"features={len(selected_indices)}"
        )

    coefficients = np.zeros(feature_matrix.shape[1], dtype=float)
    for position, feature_idx in enumerate(selected_indices):
        if feature_idx == 0:
            continue
        coefficients[feature_idx] = (
            standardized_coefficients[position] / scales[feature_idx]
        )
    intercept_position = selected_indices.index(0)
    coefficients[0] = standardized_coefficients[intercept_position] - sum(
        coefficients[feature_idx] * means[feature_idx]
        for feature_idx in selected_indices
        if feature_idx != 0
    )

    smallest_singular = float(singular_values[-1])
    condition_number = (
        None
        if smallest_singular <= 0.0
        else float(singular_values[0] / smallest_singular)
    )
    return {
        "coefficients": [float(value) for value in coefficients],
        "selected_indices": selected_indices,
        "rank": int(rank),
        "condition_number": condition_number,
    }


def _gpu_upper_tail_features(row) -> list[float]:
    """Features for a stable affine GPU latency tail above the fitted range."""
    input_scale = float(row.input_scale)
    inverse_cpu_cores = 1.0 / float(row.cpu_cores)
    inverse_cpu_cores_squared = inverse_cpu_cores**2
    return [
        1.0,
        input_scale,
        inverse_cpu_cores,
        inverse_cpu_cores_squared,
        math.log(float(row.mem_cap_gb)),
        input_scale * inverse_cpu_cores,
        input_scale * inverse_cpu_cores_squared,
    ]


def _fit_gpu_upper_tail_model(rows: list) -> dict:
    feature_matrix = np.asarray(
        [_gpu_upper_tail_features(row) for row in rows],
        dtype=float,
    )
    target = np.asarray([float(row.latency_s) for row in rows], dtype=float)
    solution = _fit_rank_revealing_linear_model(feature_matrix, target)
    selected_feature_names = [
        LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES[feature_idx]
        for feature_idx in solution["selected_indices"]
    ]
    scale_slope_features = {
        "input_scale",
        "input_scale_x_inverse_cpu_cores",
        "input_scale_x_inverse_cpu_cores_squared",
    }
    if not scale_slope_features.intersection(selected_feature_names):
        raise ValueError("GPU upper-tail model has no input-scale slope")

    coefficients = solution["coefficients"]
    training_configurations = sorted({
        (float(row.cpu_cores), float(row.mem_cap_gb)) for row in rows
    })
    fitted_slopes = []
    for cpu_cores, mem_cap_gb in training_configurations:
        at_zero = SimpleNamespace(
            input_scale=0.0,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
        )
        at_one = SimpleNamespace(
            input_scale=1.0,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
        )
        fitted_slopes.append(float(np.dot(
            coefficients,
            np.asarray(_gpu_upper_tail_features(at_one))
            - np.asarray(_gpu_upper_tail_features(at_zero)),
        )))

    return {
        "feature_names": list(LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES),
        "selected_indices": solution["selected_indices"],
        "selected_feature_names": selected_feature_names,
        "dropped_feature_names": [
            feature_name
            for feature_name in LATENCY_MODEL_GPU_UPPER_TAIL_FEATURES
            if feature_name not in selected_feature_names
        ],
        "coefficients": coefficients,
        "rank": solution["rank"],
        "condition_number": solution["condition_number"],
        "minimum_fitted_slope_s_per_scale": min(fitted_slopes),
    }


def fit_log_latency_model(
    rows: list,
    hardware_model: str,
    *,
    calibrate_gpu_upper_tail: bool = True,
) -> dict:
    """Fit log(latency) with rank-revealing least squares.

    Regressors are standardized before solving. Constant or linearly dependent
    columns are removed so CPU-only, GPU-only, and reduced resource matrices do
    not fail merely because a feature is constant.
    """
    if len(rows) < 3:
        raise ValueError(f"need at least 3 case rows, got {len(rows)}")

    input_scale_spline_knots = (
        _gpu_input_scale_spline_knots(rows)
        if hardware_model == "gpu"
        else []
    )
    feature_names = model_feature_names(
        hardware_model,
        input_scale_spline_knots,
    )
    feature_matrix = np.asarray(
        [
            _model_features(
                row,
                hardware_model,
                input_scale_spline_knots,
            )
            for row in rows
        ],
        dtype=float,
    )
    target = np.log(np.asarray([float(row.latency_s) for row in rows], dtype=float))
    solution = _fit_rank_revealing_linear_model(feature_matrix, target)
    selected_indices = solution["selected_indices"]
    model = {
        "hardware_model": hardware_model,
        "input_scale_spline_knots": input_scale_spline_knots,
        "input_scale_min": min(float(row.input_scale) for row in rows),
        "input_scale_max": max(float(row.input_scale) for row in rows),
        "feature_names": feature_names,
        "selected_indices": selected_indices,
        "selected_feature_names": [
            feature_names[feature_idx] for feature_idx in selected_indices
        ],
        "dropped_feature_names": [
            feature_names[feature_idx]
            for feature_idx in range(len(feature_names))
            if feature_idx not in selected_indices
        ],
        "coefficients": solution["coefficients"],
        "rank": solution["rank"],
        "condition_number": solution["condition_number"],
        "fit_case_rows": len(rows),
    }

    if hardware_model == "gpu" and calibrate_gpu_upper_tail:
        input_scales = sorted({float(row.input_scale) for row in rows})
        input_scale_span_ratio = input_scales[-1] / input_scales[0]
        calibration: dict[str, Any] = {
            "available": False,
            "held_out_input_scale": None,
            "mean_absolute_percentage_error": None,
            "activation_threshold_mape": (
                LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
            ),
        }
        if len(input_scales) >= 3:
            calibration_scale = input_scales[-1]
            calibration_rows = [
                row for row in rows
                if float(row.input_scale) < calibration_scale
            ]
            calibration_test_rows = [
                row for row in rows
                if float(row.input_scale) == calibration_scale
            ]
            try:
                calibration_model = fit_log_latency_model(
                    calibration_rows,
                    hardware_model,
                    calibrate_gpu_upper_tail=False,
                )
                calibration_predictions = np.asarray([
                    predict_latency(row, calibration_model)
                    for row in calibration_test_rows
                ])
                calibration_actuals = np.asarray([
                    float(row.latency_s) for row in calibration_test_rows
                ])
                calibration_mape = float(np.mean(
                    np.abs(calibration_actuals - calibration_predictions)
                    / calibration_actuals
                ))
                calibration = {
                    "available": True,
                    "held_out_input_scale": calibration_scale,
                    "train_input_scale_max": max(
                        float(row.input_scale) for row in calibration_rows
                    ),
                    "mean_absolute_percentage_error": calibration_mape,
                    "activation_threshold_mape": (
                        LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
                    ),
                }
            except ValueError as exc:
                calibration["reason"] = str(exc)

        try:
            upper_tail_model = _fit_gpu_upper_tail_model(rows)
            upper_tail_model["calibration"] = calibration
            upper_tail_model["training_input_scale_span_ratio"] = (
                input_scale_span_ratio
            )
            upper_tail_model["enabled"] = bool(
                calibration["available"]
                and calibration["mean_absolute_percentage_error"]
                > LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE
                and input_scale_span_ratio
                >= LATENCY_MODEL_GPU_UPPER_TAIL_MIN_SCALE_SPAN_RATIO
                and upper_tail_model["minimum_fitted_slope_s_per_scale"] > 0.0
            )
            model["gpu_upper_tail"] = upper_tail_model
        except ValueError as exc:
            model["gpu_upper_tail"] = {
                "enabled": False,
                "reason": str(exc),
                "calibration": calibration,
            }

    return model


def _predict_primary_latency(row, model: dict) -> float:
    """Evaluate the positive log-link model without tail extrapolation."""
    linear_prediction = sum(
        coefficient * feature
        for coefficient, feature in zip(
            model["coefficients"],
            _model_features(
                row,
                model["hardware_model"],
                model.get("input_scale_spline_knots", []),
            ),
            strict=True,
        )
    )
    # The exponential link makes predictions positive. Clipping only protects
    # artifact generation from floating-point overflow on extreme extrapolation.
    bounded_prediction = min(max(linear_prediction, -700.0), 700.0)
    return float(math.exp(bounded_prediction))


def predict_latency(row, model: dict) -> float:
    primary_prediction = _predict_primary_latency(row, model)
    upper_tail_model = model.get("gpu_upper_tail", {})
    input_scale_max = model.get("input_scale_max")
    if not (
        model.get("hardware_model") == "gpu"
        and upper_tail_model.get("enabled")
        and input_scale_max is not None
        and float(row.input_scale) > float(input_scale_max)
    ):
        return primary_prediction

    boundary_row = SimpleNamespace(
        input_scale=float(input_scale_max),
        cpu_cores=float(row.cpu_cores),
        mem_cap_gb=float(row.mem_cap_gb),
    )
    boundary_prediction = _predict_primary_latency(boundary_row, model)
    feature_delta = (
        np.asarray(_gpu_upper_tail_features(row), dtype=float)
        - np.asarray(_gpu_upper_tail_features(boundary_row), dtype=float)
    )
    tail_delta = float(np.dot(
        np.asarray(upper_tail_model["coefficients"], dtype=float),
        feature_delta,
    ))
    tail_prediction = boundary_prediction + tail_delta
    if math.isfinite(tail_prediction) and tail_prediction > 0.0:
        return tail_prediction
    return primary_prediction


def regression_metrics(
    actuals: list[float],
    predictions: list[float],
) -> dict[str, float | int | None]:
    prediction_count = len(predictions)
    nonfinite_prediction_count = sum(
        not math.isfinite(float(prediction)) for prediction in predictions
    )
    nonpositive_prediction_count = sum(
        math.isfinite(float(prediction)) and float(prediction) <= 0.0
        for prediction in predictions
    )
    valid_pairs = [
        (float(actual), float(prediction))
        for actual, prediction in zip(actuals, predictions, strict=True)
        if math.isfinite(float(actual)) and math.isfinite(float(prediction))
    ]
    if not valid_pairs:
        return {
            "r2": None,
            "mae": None,
            "rmse": None,
            "relative_mae": None,
            "mean_absolute_percentage_error": None,
            "maximum_absolute_percentage_error": None,
            "smape": None,
            "p95_absolute_error": None,
            "mean_actual": None,
            "prediction_count": prediction_count,
            "nonfinite_prediction_count": nonfinite_prediction_count,
            "nonpositive_prediction_count": nonpositive_prediction_count,
            "min_prediction": None,
            "max_prediction": None,
        }

    valid_actuals = [pair[0] for pair in valid_pairs]
    valid_predictions = [pair[1] for pair in valid_pairs]
    errors = [
        actual - predicted
        for actual, predicted in zip(valid_actuals, valid_predictions, strict=True)
    ]
    absolute_errors = [abs(error) for error in errors]
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_actual = sum(valid_actuals) / len(valid_actuals)
    ss_tot = sum((actual - mean_actual) ** 2 for actual in valid_actuals)
    ss_res = sum(error * error for error in errors)
    r2 = None if ss_tot <= 1e-24 else 1.0 - (ss_res / ss_tot)
    relative_mae = None if mean_actual <= 0.0 else mae / mean_actual
    absolute_percentage_errors = [
        abs(actual - predicted) / abs(actual)
        for actual, predicted in zip(valid_actuals, valid_predictions, strict=True)
        if abs(actual) > 0.0
    ]
    mean_absolute_percentage_error = (
        None
        if not absolute_percentage_errors
        else sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    )
    maximum_absolute_percentage_error = (
        None
        if not absolute_percentage_errors
        else max(absolute_percentage_errors)
    )
    smape_terms = [
        2.0 * abs(actual - predicted) / (abs(actual) + abs(predicted))
        for actual, predicted in zip(valid_actuals, valid_predictions, strict=True)
        if abs(actual) + abs(predicted) > 0.0
    ]
    smape = None if not smape_terms else sum(smape_terms) / len(smape_terms)
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "relative_mae": relative_mae,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "maximum_absolute_percentage_error": maximum_absolute_percentage_error,
        "smape": smape,
        "p95_absolute_error": float(np.percentile(absolute_errors, 95)),
        "mean_actual": mean_actual,
        "prediction_count": prediction_count,
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "nonpositive_prediction_count": nonpositive_prediction_count,
        "min_prediction": min(valid_predictions),
        "max_prediction": max(valid_predictions),
    }


def latency_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "latency_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required modeling columns: {sorted(missing)}")

    model_df = df[list(required)].copy()
    for col in ("cpu_cores", "mem_cap_gb", "input_scale", "latency_s"):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df[
        model_df["cpu_cores"].notna()
        & model_df["mem_cap_gb"].notna()
        & model_df["input_scale"].notna()
        & model_df["latency_s"].notna()
        & (model_df["cpu_cores"] > 0)
        & (model_df["mem_cap_gb"] > 0)
        & (model_df["input_scale"] > 0)
        & (model_df["latency_s"] > 0)
    ].copy()
    model_df["source_row"] = model_df.index
    model_df["hardware_model"] = model_df["gpu_mode"].map(_hardware_model_name)
    return model_df.sort_values(
        ["hardware_model", "cpu_cores", "mem_cap_gb", "input_scale", "source_row"]
    ).reset_index(drop=True)


def aggregate_latency_model_points(model_df: pd.DataFrame) -> list:
    """Collapse repeated windows before any fit or validation split."""
    point_df = model_df.groupby(
        ["hardware_model", "cpu_cores", "mem_cap_gb", "input_scale"],
        as_index=False,
        sort=True,
    ).agg(
        latency_s=("latency_s", "median"),
        latency_mean_s=("latency_s", "mean"),
        latency_std_s=("latency_s", lambda values: values.std(ddof=0)),
        repeat_count=("latency_s", "size"),
    )
    point_df["latency_std_s"] = point_df["latency_std_s"].fillna(0.0)
    return list(point_df.itertuples(index=False))


def point_key(row) -> tuple[str, float, float, float]:
    return (
        str(row.hardware_model),
        float(row.cpu_cores),
        float(row.mem_cap_gb),
        float(row.input_scale),
    )


def training_range(rows: list) -> dict[str, dict[str, float]]:
    return {
        "cpu_cores": {
            "min": min(float(row.cpu_cores) for row in rows),
            "max": max(float(row.cpu_cores) for row in rows),
        },
        "mem_cap_gb": {
            "min": min(float(row.mem_cap_gb) for row in rows),
            "max": max(float(row.mem_cap_gb) for row in rows),
        },
        "input_scale": {
            "min": min(float(row.input_scale) for row in rows),
            "max": max(float(row.input_scale) for row in rows),
        },
    }
