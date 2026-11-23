"""Frozen semantic components and complete-task aggregate."""
import numpy as np

def landing_status(tilt):
    if tilt is None or not np.isfinite(tilt) or not 0<=tilt<=90:return None
    return '安全着陆' if tilt<=5 else '谨慎着陆' if tilt<=15 else '不可着陆'
def proposals(values, sample_id):
    import cv2
    import numpy as np
    labels = values["semantic"]
    foreground = values["foreground_probability"]
    confidence = values["class_confidence"]
    if labels.ndim != 2 or foreground.shape != labels.shape or confidence.shape != labels.shape:
        raise ValueError("Native semantic/confidence grid mismatch")
    parts = []
    for category in range(1, 8):
        binary = ((labels == category) & np.isfinite(foreground) & (foreground >= .5)).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for number in range(1, count):
            mask = components == number
            first = int(np.flatnonzero(mask)[0])
            parts.append((first, category - 1, mask, int(stats[number, cv2.CC_STAT_AREA])))
    parts.sort(key=lambda item: item[0])
    result = []
    component_map = np.zeros(labels.shape, np.uint16)
    for index, (_, category, mask, area) in enumerate(parts):
        if area < 100:
            continue
        component_map[mask] = index + 1
        # All pixels in a semantic CC have the same argmax class, so its
        # mean class confidence equals max_c(mean_ROI p_c). No classifier.
        result.append({"object_id": f"{sample_id}:prediction:{index:04d}", "class_id": category,
            "score": float(foreground[mask].mean(dtype=np.float64)), "class_score": float(confidence[mask].mean(dtype=np.float64)),
            "component_value": index + 1, "mask": mask})
    return result, component_map

def aggregate(reports):
    counts = {key: sum(report["counts"][key] for report in reports) for key in reports[0]["counts"]}
    ng, npred = counts["gt_objects"], counts["pred_objects"]
    outcomes = [v for report in reports for v in report["per_image"][0]["gt_outcomes"]]
    errors = [v["error_deg"] if v["class_correct"] is True and v["angle_valid"] else 90. for v in outcomes]
    joint = {}
    for key in ("1", "2", "3", "5", "10"):
        tp = sum(report["joint"][key]["true_positives"] for report in reports)
        joint[key] = {"true_positives": tp, "false_positives": npred - tp, "false_negatives": ng - tp,
            "precision": tp / npred if npred else None, "recall": tp / ng if ng else None,
            "f1": 2 * tp / (ng + npred) if ng + npred else None}
    return {"images": len(reports), "NG": ng, "NP": npred, "counts": counts, "joint": joint,
        "MAE90": sum(errors) / ng if ng else None, "failure90_count": sum(not (v["class_correct"] is True and v["angle_valid"]) for v in outcomes),
        "denominator": "all_GT_objects; wrong_class_missed_invalid=90; FP_in_joint_F1"}
