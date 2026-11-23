"""Read-only, object-level evaluation of cooperative surface inclination.

Records are image dictionaries with ``sample_id`` and an ``objects`` list.
Every object needs a unique nonempty string ``object_id``, a 2-D binary ``mask``
and, for evaluation, a nonnegative integer or nonempty string ``class_id``.
Masks must share an image's pixel grid. Boolean or numeric 0/1 masks are accepted;
this module deliberately does not infer a threshold for arbitrary mask images.

Ground truth must already belong to the independently qualified main track:
``eligible=False``, an empty GT mask or missing/invalid GT angle is an error,
never an instruction to silently shrink the denominator. Prediction angles that
are missing, nonfinite, outside [0, 90], or declared invalid are recorded as
failures and retained in the *base* prediction denominator. Malformed structure,
IDs, classes and scores raise ValueError with their field context.

``confidence_threshold=None`` performs base evaluation, retaining all submitted
predictions. A numerical threshold performs selective evaluation: only valid
angle predictions with score >= threshold are accepted, and spatial matching is
recomputed on that subset. Submitted/rejected counts remain visible. ``score``
defaults to 1 and ``angle_valid`` defaults to True; scores are finite in [0, 1].
Results contain only JSON-compatible scalars/containers; unavailable metrics are
None. No input, annotation, cache or output file is modified.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["match_masks", "evaluate_predictions", "evaluate_selective_predictions"]


def _sequence(value, field):
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{field} must be a sequence")
    try:
        return list(value)
    except TypeError as error:
        raise ValueError(f"{field} must be a sequence") from error


def _identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _number(value, field, minimum, maximum, *, minimum_inclusive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    in_range = minimum <= result <= maximum if minimum_inclusive else minimum < result <= maximum
    if not np.isfinite(result) or not in_range:
        left = "[" if minimum_inclusive else "("
        raise ValueError(f"{field} must lie in {left}{minimum}, {maximum}]")
    return result


def _boolean(value, field):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be boolean")
    return bool(value)


def _mask(value, field):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a rectangular binary 2-D mask") from error
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{field} must have a nonempty 2-D shape")
    if array.dtype.kind not in "buif":
        raise ValueError(f"{field} must contain booleans or numeric 0/1 values")
    if array.dtype.kind != "b" and not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{field} must contain only binary 0/1 values")
    return array.astype(bool, copy=False)


def _mask_objects(objects, context):
    result, ids = [], set()
    for index, original in enumerate(_sequence(objects, context)):
        prefix = f"{context}[{index}]"
        if not isinstance(original, Mapping):
            raise ValueError(f"{prefix} must be an object mapping")
        object_id = _identifier(original.get("object_id"), f"{prefix}.object_id")
        if object_id in ids:
            raise ValueError(f"Duplicate object_id {object_id!r} in {context}")
        ids.add(object_id)
        result.append({"object_id": object_id, "mask": _mask(original.get("mask"), f"{prefix}.mask")})
    _check_shapes(result, context)
    return result


def _check_shapes(objects, context):
    shapes = {row["mask"].shape for row in objects}
    if len(shapes) > 1:
        raise ValueError(f"{context} mask shapes differ: {sorted(shapes)}")


def _match_normalized(gt, pred, iou_threshold):
    """Hungarian assignment with a strictly dominant cardinality bonus.

    IDs are sorted before invoking the deterministic solver, so input-list order
    cannot choose an equal-cost solution. No epsilon is added to IoU: such a
    perturbation could reverse a genuinely better overlap total. Empty masks
    have zero overlap even when both masks are empty.
    """
    _check_shapes(gt + pred, "GT/prediction")
    if not gt or not pred:
        return []
    g_order = sorted(range(len(gt)), key=lambda i: gt[i]["object_id"])
    p_order = sorted(range(len(pred)), key=lambda i: pred[i]["object_id"])
    ious = np.zeros((len(gt), len(pred)), dtype=np.float64)
    for gi, original_g in enumerate(g_order):
        g_mask = gt[original_g]["mask"]
        for pi, original_p in enumerate(p_order):
            p_mask = pred[original_p]["mask"]
            intersection = int(np.count_nonzero(g_mask & p_mask))
            union = int(np.count_nonzero(g_mask | p_mask))
            ious[gi, pi] = intersection / union if union else 0.0
    eligible = ious >= iou_threshold
    # One additional eligible edge outweighs every possible change to the IoU
    # sum (at most min(NG, NP)). Each GT can instead use a zero-valued dummy.
    bonus = min(len(gt), len(pred)) + 1.0
    reward = np.zeros((len(gt), len(pred) + len(gt)), dtype=np.float64)
    reward[:, :len(pred)] = np.where(eligible, bonus + ious, -bonus)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    return [(g_order[int(g)], p_order[int(p)], float(ious[g, p]))
            for g, p in zip(rows, columns)
            if p < len(pred) and eligible[g, p]]


def match_masks(gt_objects, pred_objects, iou_threshold=0.5):
    """Return ``(original_gt_index, original_pred_index, IoU)`` matches.

    Eligible edges satisfy IoU >= threshold. Optimize match count first, then
    summed IoU. Stable object-ID ordering resolves equal-cost solver ties,
    independent of input permutations. Classes, angles, scores and eligibility
    flags do not participate in this spatial operation. Only object_id and mask
    are required by this function. Every supplied mask is validated, including
    when the opposite object list is empty.
    """
    threshold = _number(iou_threshold, "iou_threshold", 0, 1, minimum_inclusive=False)
    gt = _mask_objects(gt_objects, "gt_objects")
    pred = _mask_objects(pred_objects, "pred_objects")
    return _match_normalized(gt, pred, threshold)


def _class_id(value, field):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a nonnegative integer or nonempty string")
    if isinstance(value, Integral) and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{field} must be a nonnegative integer or nonempty string")


def _angle(original, prefix, is_gt):
    declared_valid = _boolean(original.get("angle_valid", True), f"{prefix}.angle_valid")
    value = original.get("angle_deg")
    if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)):
        raise ValueError(f"{prefix}.angle_deg must be numeric or None")
    value = float(value) if value is not None else None
    reason = None
    if value is None:
        reason = "missing_angle"
    elif not np.isfinite(value):
        reason, value = "non_finite_angle", None
    elif not 0 <= value <= 90:
        reason, value = "out_of_range_angle", None
    elif not declared_valid:
        reason = "declared_invalid"
    if is_gt and reason is not None:
        raise ValueError(f"{prefix}.angle_deg/angle_valid is unqualified ground truth: {reason}")
    return value, reason is None, reason


def _records(records, context, *, is_gt):
    result = {}
    for index, record in enumerate(_sequence(records, context)):
        prefix = f"{context}[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{prefix} must be an image record mapping")
        sample_id = _identifier(record.get("sample_id"), f"{prefix}.sample_id")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {context}")
        originals = _sequence(record.get("objects"), f"{prefix}.objects")
        objects = _mask_objects(originals, f"{prefix}.objects")
        for original, row in zip(originals, objects):
            object_prefix = f"{prefix}.objects[{row['object_id']!r}]"
            if "eligible" in original:
                eligible = _boolean(original["eligible"], f"{object_prefix}.eligible")
                if is_gt and not eligible:
                    raise ValueError(f"{object_prefix}.eligible=False: main-track GT must be prequalified")
            if is_gt and not row["mask"].any():
                raise ValueError(f"{object_prefix}.mask is empty ground-truth annotation")
            row["class_id"] = _class_id(original.get("class_id"), f"{object_prefix}.class_id")
            value, valid, reason = _angle(original, object_prefix, is_gt)
            row.update(angle_deg=value, angle_valid=valid, angle_invalid_reason=reason,
                       score=_number(original.get("score", 1.0), f"{object_prefix}.score", 0, 1))
        result[sample_id] = sorted(objects, key=lambda row: row["object_id"])
    return result


def _thresholds(values, field, minimum, maximum):
    result = [_number(value, field, minimum, maximum) for value in _sequence(values, field)]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field} must be a nonempty sequence of unique numbers")
    return result


def _threshold_key(value):
    return str(int(value)) if value.is_integer() else repr(value)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _rates(true_positives, n_gt, n_pred):
    return {"true_positives": int(true_positives),
            "false_positives": int(n_pred - true_positives),
            "false_negatives": int(n_gt - true_positives),
            "precision": _ratio(true_positives, n_pred),
            "recall": _ratio(true_positives, n_gt),
            "f1": _ratio(2 * true_positives, n_gt + n_pred)}


def _error_stats(errors, n_gt):
    values = np.asarray(errors, dtype=np.float64)
    return {"denominator": len(errors), "gt_coverage": _ratio(len(errors), n_gt),
            "mae_deg": float(values.mean()) if len(errors) else None,
            "rmse_deg": float(np.sqrt(np.mean(values ** 2))) if len(errors) else None,
            "median_deg": float(np.median(values)) if len(errors) else None,
            "p95_deg": float(np.percentile(values, 95, method="linear")) if len(errors) else None}


def _empty_counts():
    return {key: 0 for key in (
        "images", "gt_objects", "pred_objects", "submitted_pred_objects", "rejected_predictions",
        "localization_matches", "missed_gt", "unmatched_predictions", "wrong_class",
        "prediction_angle_invalid", "submitted_prediction_angle_invalid", "matched_angle_invalid",
        "matched_angle_valid", "matched_class_correct", "matched_class_correct_angle_valid",
    )}


def _evaluate(gt_records, pred_records, angle_thresholds, iou_threshold, confidence_threshold):
    counts = _empty_counts()
    errors, correct_errors, per_image = [], [], []
    successes = {key: 0 for key in angle_thresholds}
    for sample_id in sorted(set(gt_records) | set(pred_records)):
        gt = gt_records.get(sample_id, [])
        submitted = pred_records.get(sample_id, [])
        # Validate even rejected predictions against the correct image grid.
        _check_shapes(gt + submitted, f"sample_id={sample_id!r}")
        pred = submitted if confidence_threshold is None else [
            row for row in submitted if row["angle_valid"] and row["score"] >= confidence_threshold]
        matching = _match_normalized(gt, pred, iou_threshold)
        by_gt = {g: (p, iou) for g, p, iou in matching}
        matched_predictions = {p for _, p, _ in matching}
        image_counts = _empty_counts()
        image_counts.update(images=1, gt_objects=len(gt), pred_objects=len(pred),
                            submitted_pred_objects=len(submitted), rejected_predictions=len(submitted) - len(pred),
                            localization_matches=len(matching), missed_gt=len(gt) - len(matching),
                            unmatched_predictions=len(pred) - len(matching),
                            prediction_angle_invalid=sum(not row["angle_valid"] for row in pred),
                            submitted_prediction_angle_invalid=sum(not row["angle_valid"] for row in submitted))
        matches, gt_outcomes, image_errors, image_correct_errors = [], [], [], []
        image_successes = {key: 0 for key in angle_thresholds}
        for gi, ground_truth in enumerate(gt):
            outcome = {"gt_object_id": ground_truth["object_id"], "gt_class_id": ground_truth["class_id"],
                       "gt_angle_deg": ground_truth["angle_deg"], "pred_object_id": None,
                       "pred_class_id": None, "pred_angle_deg": None, "iou": None,
                       "class_correct": None, "angle_valid": False, "angle_invalid_reason": None,
                       "error_deg": None, "failure_reasons": ["missed"],
                       "joint_success": {_threshold_key(t): False for t in angle_thresholds}}
            if gi in by_gt:
                pi, iou = by_gt[gi]
                prediction = pred[pi]
                class_correct = prediction["class_id"] == ground_truth["class_id"]
                valid = prediction["angle_valid"]
                error = abs(prediction["angle_deg"] - ground_truth["angle_deg"]) if valid else None
                image_counts["wrong_class"] += int(not class_correct)
                image_counts["matched_class_correct"] += int(class_correct)
                image_counts["matched_angle_invalid"] += int(not valid)
                image_counts["matched_angle_valid"] += int(valid)
                image_counts["matched_class_correct_angle_valid"] += int(class_correct and valid)
                if valid:
                    image_errors.append(error)
                    if class_correct:
                        image_correct_errors.append(error)
                flags = {_threshold_key(t): bool(class_correct and valid and error <= t) for t in angle_thresholds}
                for t in angle_thresholds:
                    image_successes[t] += int(flags[_threshold_key(t)])
                reasons = (["wrong_class"] if not class_correct else []) + (["angle_invalid"] if not valid else [])
                outcome.update(pred_object_id=prediction["object_id"], pred_class_id=prediction["class_id"],
                               pred_angle_deg=prediction["angle_deg"], iou=iou,
                               class_correct=bool(class_correct), angle_valid=bool(valid),
                               angle_invalid_reason=prediction["angle_invalid_reason"], error_deg=error,
                               failure_reasons=reasons, joint_success=flags)
                matches.append(dict(outcome))
            gt_outcomes.append(outcome)
        unmatched = [{"pred_object_id": row["object_id"], "pred_class_id": row["class_id"],
                      "pred_angle_deg": row["angle_deg"], "angle_valid": row["angle_valid"],
                      "angle_invalid_reason": row["angle_invalid_reason"], "score": row["score"]}
                     for i, row in enumerate(pred) if i not in matched_predictions]
        per_image.append({"sample_id": sample_id, "NG": len(gt), "NP": len(pred), "counts": image_counts,
                          "matches": matches, "gt_outcomes": gt_outcomes,
                          "missed_gt_ids": [row["gt_object_id"] for row in gt_outcomes if row["pred_object_id"] is None],
                          "unmatched_predictions": unmatched,
                          "joint": {_threshold_key(t): {"angle_threshold_deg": t, **_rates(image_successes[t], len(gt), len(pred))}
                                    for t in angle_thresholds}})
        for name in counts:
            counts[name] += image_counts[name]
        for t in angle_thresholds:
            successes[t] += image_successes[t]
        errors.extend(image_errors)
        correct_errors.extend(image_correct_errors)
    n_gt, n_pred = counts["gt_objects"], counts["pred_objects"]
    return {"schema_version": 1, "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold, "angle_thresholds_deg": angle_thresholds,
            "NG": n_gt, "NP": n_pred, "counts": counts,
            "localization": _rates(counts["localization_matches"], n_gt, n_pred),
            "joint": {_threshold_key(t): {"angle_threshold_deg": t, **_rates(successes[t], n_gt, n_pred)}
                      for t in angle_thresholds},
            "conditional_angle": _error_stats(errors, n_gt),
            "correct_class_conditional_angle": _error_stats(correct_errors, n_gt),
            "valid_target_coverage": _ratio(len(correct_errors), n_gt),
            "prediction_acceptance_rate": _ratio(n_pred, counts["submitted_pred_objects"]),
            "per_image": per_image}


def evaluate_predictions(gt_records, pred_records, angle_thresholds=(2, 5, 10),
                         iou_threshold=0.5, confidence_threshold=None):
    """Evaluate the union of image IDs, retaining misses, duplicates and errors.

    ``conditional_angle`` covers *all* spatially matched valid angles, including
    wrong classes, and always gives its denominator. The class-correct subset
    is separate. Joint success requires spatial match, correct class and valid
    absolute angle error <= each angle threshold. Joint precision uses NP,
    recall uses NG and F1 is 2*JTP/(NP+NG). A zero individual denominator is None;
    if NG=0 and NP>0, joint precision/F1 are 0 and recall is None. If both are
    zero all three are None. No failed prediction is imputed an angle of zero.
    """
    iou = _number(iou_threshold, "iou_threshold", 0, 1, minimum_inclusive=False)
    thresholds = _thresholds(angle_thresholds, "angle_thresholds", 0, 90)
    confidence = None if confidence_threshold is None else _number(
        confidence_threshold, "confidence_threshold", 0, 1)
    gt = _records(gt_records, "gt_records", is_gt=True)
    pred = _records(pred_records, "pred_records", is_gt=False)
    return _evaluate(gt, pred, thresholds, iou, confidence)


def evaluate_selective_predictions(gt_records, pred_records,
                                  confidence_thresholds=(0, 0.25, 0.5, 0.75, 1),
                                  angle_thresholds=(2, 5, 10), iou_threshold=0.5):
    """Return evaluations for each acceptance threshold, in supplied order.

    Match independently inside each accepted subset; a rejected low-score
    proposal can never continue occupying a GT assignment. Normalizing once
    also permits one-shot iterable record inputs without consuming them twice.
    """
    iou = _number(iou_threshold, "iou_threshold", 0, 1, minimum_inclusive=False)
    thresholds = _thresholds(angle_thresholds, "angle_thresholds", 0, 90)
    confidences = _thresholds(confidence_thresholds, "confidence_thresholds", 0, 1)
    gt = _records(gt_records, "gt_records", is_gt=True)
    pred = _records(pred_records, "pred_records", is_gt=False)
    return {"schema_version": 1, "confidence_thresholds": confidences,
            "evaluations": [_evaluate(gt, pred, thresholds, iou, threshold) for threshold in confidences]}
