"""Target-free MoGe-3 ROI descriptors for a frozen RGB/geometry fusion head.

Inputs are native camera fields, not crop-resized fields: axial Z in metres,
OpenCV camera normals, a visible/predicted semantic mask, pixel K and gravity.
The descriptor does not open files or targets, initialize CUDA, fit any learned
normalization, infer missing geometry, or align depth scale with ground truth.
"""
from __future__ import annotations

import numpy as np


SCHEMA = "colsi_strong_moge_native_ROI48_v1"
MAX_POINTS = 1024
MIN_SUPPORT = 32
DESCRIPTOR_NAMES = (
    "roi_fraction", "bbox_width_fraction", "bbox_height_fraction",
    "centroid_x_fraction", "centroid_y_fraction", "depth_coverage",
    "normal_coverage", "log1p_roi_pixels", "depth_valid",
    "log1p_z_p10_m", "log1p_z_median_m", "log1p_z_p90_m",
    "depth_std_over_mean", "depth_iqr_over_median", "plane_valid",
    "plane_nx", "plane_ny", "plane_nz", "plane_tilt_over90",
    "plane_eig0_over_sum", "plane_eig1_over_eig2",
    "plane_residual_median_over_extent", "plane_residual_p90_over_extent",
    "log1p_plane_extent_m", "plane_inlier_fraction_004_medianZ",
    "direct_normal_valid", "direct_nx", "direct_ny", "direct_nz",
    "direct_tilt_over90", "normal_concentration", "normal_eigen_gap",
    "normal_deviation_median_over90", "normal_deviation_p90_over90",
    "pixel_tilt_median_over90", "pixel_tilt_std_over90",
    "normal_within15deg_fraction", "plane_direct_disagreement_over90",
    "normal_plane_disagreement_valid", "sampled_point_fraction",
    "sampled_normal_fraction", "fx_over_width", "fy_over_height",
    "cx_over_width", "cy_over_height", "gravity_x", "gravity_y", "gravity_z",
)
INDEX = {name: i for i, name in enumerate(DESCRIPTOR_NAMES)}
VALIDITY_NAMES = ("depth_valid", "plane_valid", "direct_normal_valid",
                  "normal_plane_disagreement_valid")


def _unit(vector):
    value = np.asarray(vector, np.float64)
    length = np.linalg.norm(value)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.isfinite(length) or length < 1e-12:
        raise ValueError("Finite nonzero camera three-vector required")
    return value / length


def _tilt(normal, reference):
    """Unsigned [0,90] angle, stable also at horizontal surfaces."""
    n, g = _unit(normal), _unit(reference)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n, g)), abs(n @ g))))


def _canonical_normal(vector):
    n = _unit(vector)
    # Camera-facing convention is deterministic; the estimated angle is axial.
    pivot = 2 if abs(n[2]) > 1e-12 else int(np.argmax(np.abs(n)))
    return -n if n[pivot] < 0 else n


def extract_descriptor(depth_z, normals, valid_depth, valid_normal, roi_mask, K, gravity):
    """Return finite48 values and explicit validity for this exact ROI.

    Sampling is deterministic raster-order linspace, selected BEFORE geometry
    validity, at most1024 pixels. Coverage is computed over all ROI pixels.
    Missing representations may be None; supplied fields require their explicit
    validity mask. Numerical zero placeholders always have accompanying flags.
    Normalization must be fitted externally on training descriptors only.
    """
    mask = np.asarray(roi_mask)
    if mask.ndim != 2 or mask.dtype != bool or min(mask.shape) < 1:
        raise ValueError("Explicit boolean native[H,W] ROI mask required")
    h, w = mask.shape
    intrinsic = np.asarray(K, np.float64)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-10)):
        raise ValueError("Finite positive-focal pinhole K required")
    try:
        inverse_K = np.linalg.inv(intrinsic)
    except np.linalg.LinAlgError as error:
        raise ValueError("Invertible K required") from error
    g = _unit(gravity)
    output = np.zeros(len(DESCRIPTOR_NAMES), np.float64)

    def put(name, value):
        output[INDEX[name]] = float(value)

    put("fx_over_width", intrinsic[0, 0] / w)
    put("fy_over_height", intrinsic[1, 1] / h)
    put("cx_over_width", intrinsic[0, 2] / w)
    put("cy_over_height", intrinsic[1, 2] / h)
    for name, value in zip(("gravity_x", "gravity_y", "gravity_z"), g):
        put(name, value)

    z = None if depth_z is None else np.asarray(depth_z)
    nfield = None if normals is None else np.asarray(normals)
    if z is not None:
        if z.shape != mask.shape or valid_depth is None:
            raise ValueError("Depth and explicit depth-validity must match native ROI")
        supplied_zvalid = np.asarray(valid_depth)
        if supplied_zvalid.shape != mask.shape or supplied_zvalid.dtype != bool:
            raise ValueError("Explicit boolean native depth validity required")
        zvalid = np.zeros_like(mask)
        roi_z = z[mask]
        zvalid[mask] = supplied_zvalid[mask] & np.isfinite(roi_z) & (roi_z > 0)
    else:
        if valid_depth is not None:
            raise ValueError("Depth-validity cannot be supplied without depth")
        zvalid = np.zeros_like(mask)
    if nfield is not None:
        if nfield.shape != (h, w, 3) or valid_normal is None:
            raise ValueError("Normals and explicit normal-validity must match native ROI")
        supplied_nvalid = np.asarray(valid_normal)
        if supplied_nvalid.shape != mask.shape or supplied_nvalid.dtype != bool:
            raise ValueError("Explicit boolean native normal validity required")
        nvalid = np.zeros_like(mask)
        roi_normals = nfield[mask]
        lengths = np.linalg.norm(roi_normals, axis=-1)
        nvalid[mask] = (supplied_nvalid[mask] & np.isfinite(roi_normals).all(-1)
                        & np.isfinite(lengths) & (lengths > 1e-8))
    else:
        if valid_normal is not None:
            raise ValueError("Normal-validity cannot be supplied without normals")
        nvalid = np.zeros_like(mask)

    flat = np.flatnonzero(mask)
    roi_count = len(flat)
    diagnostics = {"schema": SCHEMA, "native_hw": [h, w], "roi_pixels": roi_count,
                   "depth_valid_pixels": int(np.count_nonzero(mask & zvalid)),
                   "normal_valid_pixels": int(np.count_nonzero(mask & nvalid)),
                   "sampled_pixels": min(roi_count, MAX_POINTS), "max_points": MAX_POINTS,
                   "min_support": MIN_SUPPORT, "GT_used": False,
                   "depth_quantity": "axial_Z_metres_no_GT_scale_alignment",
                   "normal_frame": "OpenCV_camera_xright_ydown_zforward_axial",
                   "plane_method": "unweighted_orthogonal_TLS_numpy_eigh",
                   "normal_method": "sign_invariant_second_moment_numpy_eigh",
                   "sampling": "raster_linspace_endpoints_before_validity_max1024",
                   "invalid_reasons": []}
    plane_n = normal_n = None
    if roi_count:
        ys, xs = np.divmod(flat, w)
        put("roi_fraction", roi_count / (h * w))
        put("bbox_width_fraction", (xs.max() - xs.min() + 1) / w)
        put("bbox_height_fraction", (ys.max() - ys.min() + 1) / h)
        put("centroid_x_fraction", (xs.mean() + .5) / w)
        put("centroid_y_fraction", (ys.mean() + .5) / h)
        put("log1p_roi_pixels", np.log1p(roi_count))
        put("depth_coverage", diagnostics["depth_valid_pixels"] / roi_count)
        put("normal_coverage", diagnostics["normal_valid_pixels"] / roi_count)
        sampled = flat if roi_count <= MAX_POINTS else flat[np.linspace(0, roi_count - 1, MAX_POINTS, dtype=np.int64)]
        sy, sx = np.divmod(sampled, w)
        good_z = zvalid[sy, sx]
        good_n = nvalid[sy, sx]
        diagnostics["sampled_valid_depth"] = int(good_z.sum())
        diagnostics["sampled_valid_normal"] = int(good_n.sum())
        put("sampled_point_fraction", good_z.sum() / MAX_POINTS)
        put("sampled_normal_fraction", good_n.sum() / MAX_POINTS)
        if good_z.sum() >= MIN_SUPPORT:
            depth = np.asarray(z[sy[good_z], sx[good_z]], np.float64)
            q10, q25, q50, q75, q90 = np.quantile(depth, [.1, .25, .5, .75, .9])
            put("depth_valid", 1)
            for name, value in zip(("log1p_z_p10_m", "log1p_z_median_m", "log1p_z_p90_m"), (q10, q50, q90)):
                put(name, np.log1p(value))
            put("depth_std_over_mean", depth.std() / max(depth.mean(), 1e-12))
            put("depth_iqr_over_median", (q75 - q25) / max(q50, 1e-12))
            rays = np.column_stack((sx[good_z], sy[good_z], np.ones(good_z.sum()))) @ inverse_K.T
            points = rays * depth[:, None]
            center = points.mean(0)
            centered = points - center
            eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
            eigenvalues = np.maximum(eigenvalues, 0)
            if eigenvalues[2] > 1e-12 and eigenvalues[1] / eigenvalues[2] > 1e-4:
                plane_n = _canonical_normal(eigenvectors[:, 0])
                residual = np.abs(centered @ plane_n)
                extent = float(np.sqrt(eigenvalues[1:].sum()))
                put("plane_valid", 1)
                for name, value in zip(("plane_nx", "plane_ny", "plane_nz"), plane_n):
                    put(name, value)
                put("plane_tilt_over90", _tilt(plane_n, g) / 90)
                put("plane_eig0_over_sum", eigenvalues[0] / max(eigenvalues.sum(), 1e-20))
                put("plane_eig1_over_eig2", eigenvalues[1] / eigenvalues[2])
                put("plane_residual_median_over_extent", np.median(residual) / max(extent, 1e-12))
                put("plane_residual_p90_over_extent", np.quantile(residual, .9) / max(extent, 1e-12))
                put("log1p_plane_extent_m", np.log1p(extent))
                put("plane_inlier_fraction_004_medianZ", np.mean(residual <= max(.004 * q50, 1e-6)))
            else:
                diagnostics["invalid_reasons"].append("plane_spatially_degenerate")
        else:
            diagnostics["invalid_reasons"].append("insufficient_sampled_positive_depth")
        if good_n.sum() >= MIN_SUPPORT:
            vectors = np.asarray(nfield[sy[good_n], sx[good_n]], np.float64)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            eigenvalues, eigenvectors = np.linalg.eigh(vectors.T @ vectors / len(vectors))
            denominator = max(float(eigenvalues.sum()), 1e-20)
            concentration = float(eigenvalues[-1] / denominator)
            gap = float((eigenvalues[-1] - eigenvalues[-2]) / denominator)
            put("normal_concentration", concentration)
            put("normal_eigen_gap", gap)
            if gap > 1e-4:
                normal_n = _canonical_normal(eigenvectors[:, -1])
                angular = np.degrees(np.arccos(np.clip(np.abs(vectors @ normal_n), 0, 1)))
                pixel_tilt = np.degrees(np.arccos(np.clip(np.abs(vectors @ g), 0, 1)))
                put("direct_normal_valid", 1)
                for name, value in zip(("direct_nx", "direct_ny", "direct_nz"), normal_n):
                    put(name, value)
                put("direct_tilt_over90", _tilt(normal_n, g) / 90)
                put("normal_deviation_median_over90", np.median(angular) / 90)
                put("normal_deviation_p90_over90", np.quantile(angular, .9) / 90)
                put("pixel_tilt_median_over90", np.median(pixel_tilt) / 90)
                put("pixel_tilt_std_over90", pixel_tilt.std() / 90)
                put("normal_within15deg_fraction", np.mean(angular <= 15))
            else:
                diagnostics["invalid_reasons"].append("normal_axis_degenerate")
        else:
            diagnostics["invalid_reasons"].append("insufficient_sampled_normals")
    else:
        diagnostics["invalid_reasons"].append("empty_ROI")
        diagnostics.update(sampled_valid_depth=0, sampled_valid_normal=0)
    if plane_n is not None and normal_n is not None:
        put("normal_plane_disagreement_valid", 1)
        put("plane_direct_disagreement_over90", _tilt(plane_n, normal_n) / 90)
    if not np.isfinite(output).all():
        raise ValueError("Descriptor arithmetic produced nonfinite values; do not silently repair")
    values = output.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor exceeds float32 range")
    return {"values": values, "names": DESCRIPTOR_NAMES,
            "geometry_valid": bool(plane_n is not None or normal_n is not None),
            "diagnostics": diagnostics}

