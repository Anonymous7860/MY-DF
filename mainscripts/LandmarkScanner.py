"""
Bad Landmark Scanner & Fixer for DeepFaceLab
=============================================
Scans aligned facesets for faces with problematic 68-point landmarks,
reports findings, and optionally re-extracts landmarks using FAN
with two-pass S3FD refinement.
"""
import math
import traceback
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.imagelib import estimate_sharpness
from DFLIMG import *
from facelib import LandmarksProcessor, FaceType


# ─── Heuristic thresholds ─────────────────────────────────────────
OUT_OF_BOUNDS_MARGIN   = 0          # pixels — any point outside image
COLLAPSED_RATIO        = 0.10       # lmrk bbox < 10% of image → collapsed
EYE_ASPECT_MIN         = 0.05       # eye height/width minimum
JAW_ANGLE_THRESHOLD    = 45.0       # degrees — max curvature jump in jaw
ASYMMETRY_RATIO        = 3.0        # max deviation from template symmetry
XSEG_HULL_MISMATCH     = 0.60       # XSeg mask area / hull area threshold
DUPLICATE_DIST_MIN     = 1.0        # px — min distance between non-adjacent points
NOSE_BRIDGE_DEVIATION  = 0.35       # max horizontal zigzag as fraction of eye dist
SOURCE_RECT_MIN_PX     = 60         # source face detection rect minimum size
BLUR_THRESHOLD         = 10.0       # sharpness below this = unreliable landmarks

# Yaw tiers (radians)
YAW_MILD     = math.radians(30)
YAW_EXTREME  = math.radians(60)


# ─── Individual checks ───────────────────────────────────────────

def _check_nan_inf(landmarks):
    """Check for NaN or Inf values in landmarks."""
    if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
        return "NaN/Inf values in landmarks"
    return None


def _check_out_of_bounds(landmarks, image_size):
    """Check if any landmark point is outside the image bounds."""
    if np.any(landmarks < -OUT_OF_BOUNDS_MARGIN) or np.any(landmarks > image_size + OUT_OF_BOUNDS_MARGIN):
        n_bad = int(np.sum(
            (landmarks[:, 0] < -OUT_OF_BOUNDS_MARGIN) |
            (landmarks[:, 0] > image_size + OUT_OF_BOUNDS_MARGIN) |
            (landmarks[:, 1] < -OUT_OF_BOUNDS_MARGIN) |
            (landmarks[:, 1] > image_size + OUT_OF_BOUNDS_MARGIN)
        ))
        return f"{n_bad} point(s) out of bounds"
    return None


def _check_collapsed(landmarks, image_size):
    """Check if all landmarks are collapsed into a tiny region."""
    bbox_w = landmarks[:, 0].max() - landmarks[:, 0].min()
    bbox_h = landmarks[:, 1].max() - landmarks[:, 1].min()
    if bbox_w < image_size * COLLAPSED_RATIO and bbox_h < image_size * COLLAPSED_RATIO:
        return f"Collapsed face (bbox {bbox_w:.0f}x{bbox_h:.0f} vs image {image_size})"
    return None


def _eye_aspect_ratio(eye_pts):
    """Compute height/width ratio for a 6-point eye (36-41 or 42-47)."""
    width = npla.norm(eye_pts[3] - eye_pts[0])
    if width < 1e-6:
        return 0.0
    h1 = npla.norm(eye_pts[1] - eye_pts[5])
    h2 = npla.norm(eye_pts[2] - eye_pts[4])
    return (h1 + h2) / (2.0 * width)


def _check_eye_aspect(landmarks, abs_yaw):
    """Check if eyes are collapsed to lines. Yaw-aware."""
    r_eye = landmarks[36:42]  # right eye
    l_eye = landmarks[42:48]  # left eye
    r_ear = _eye_aspect_ratio(r_eye)
    l_ear = _eye_aspect_ratio(l_eye)

    if abs_yaw > YAW_EXTREME:
        # Extreme profile: only check the more-visible eye
        # Positive yaw = face turned right → left eye more visible
        # Negative yaw = face turned left → right eye more visible
        # We use abs_yaw so just check the larger EAR (more-visible eye)
        ear = max(r_ear, l_ear)
        if ear < EYE_ASPECT_MIN:
            return f"Visible eye collapsed (EAR={ear:.3f})"
    elif abs_yaw > YAW_MILD:
        # Moderate yaw: use relaxed threshold
        if r_ear < EYE_ASPECT_MIN * 0.5 and l_ear < EYE_ASPECT_MIN * 0.5:
            return f"Both eyes collapsed (R={r_ear:.3f}, L={l_ear:.3f})"
    else:
        # Frontal: both eyes must be reasonable
        if r_ear < EYE_ASPECT_MIN or l_ear < EYE_ASPECT_MIN:
            return f"Eye collapsed (R_EAR={r_ear:.3f}, L_EAR={l_ear:.3f})"
    return None


def _check_asymmetry(landmarks, abs_yaw):
    """Check left-right symmetry relative to nose midline. Yaw-aware."""
    if abs_yaw > YAW_EXTREME:
        return None  # skip for extreme profiles

    nose_mid = landmarks[30]  # tip of nose
    r_eye_center = landmarks[36:42].mean(axis=0)
    l_eye_center = landmarks[42:48].mean(axis=0)

    dist_r = abs(r_eye_center[0] - nose_mid[0])
    dist_l = abs(l_eye_center[0] - nose_mid[0])

    if dist_r < 1e-6 or dist_l < 1e-6:
        return "Eye center on nose midline (degenerate)"

    ratio = max(dist_r, dist_l) / min(dist_r, dist_l)
    threshold = ASYMMETRY_RATIO * (2.0 if abs_yaw > YAW_MILD else 1.0)

    if ratio > threshold:
        return f"Asymmetric eyes (ratio={ratio:.1f}, threshold={threshold:.1f})"
    return None


def _angle_between_vectors(v1, v2):
    """Angle in degrees between two 2D vectors."""
    cos_a = np.dot(v1, v2) / (npla.norm(v1) * npla.norm(v2) + 1e-8)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def _check_jaw_self_intersection(landmarks, abs_yaw):
    """Check if jaw contour has sharp kinks (self-intersection proxy). Yaw-aware."""
    jaw = landmarks[0:17]  # 17 jaw points

    threshold = JAW_ANGLE_THRESHOLD
    if abs_yaw > YAW_EXTREME:
        threshold *= 2.0  # much more lenient for extreme profiles
    elif abs_yaw > YAW_MILD:
        threshold *= 1.5

    for i in range(1, len(jaw) - 1):
        v1 = jaw[i] - jaw[i - 1]
        v2 = jaw[i + 1] - jaw[i]
        if npla.norm(v1) < 1e-6 or npla.norm(v2) < 1e-6:
            continue
        angle = _angle_between_vectors(v1, v2)
        deviation = abs(180.0 - angle)
        if deviation > threshold:
            return f"Jaw kink at point {i} (deviation={deviation:.1f}°)"
    return None


def _check_jaw_smoothness(landmarks, abs_yaw):
    """Check jaw contour for occlusion-caused bumps by analyzing curvature changes."""
    jaw = landmarks[0:17]

    threshold = JAW_ANGLE_THRESHOLD
    if abs_yaw > YAW_EXTREME:
        threshold *= 2.5
    elif abs_yaw > YAW_MILD:
        threshold *= 1.5

    # Compute angles between consecutive segments
    angles = []
    for i in range(1, len(jaw) - 1):
        v1 = jaw[i] - jaw[i - 1]
        v2 = jaw[i + 1] - jaw[i]
        n1 = npla.norm(v1)
        n2 = npla.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(180.0)
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(math.degrees(math.acos(cos_a)))

    if len(angles) < 3:
        return None

    # Look for sudden curvature changes (occlusion signature)
    for i in range(1, len(angles)):
        change = abs(angles[i] - angles[i - 1])
        if change > threshold:
            return f"Occlusion-like jaw bump near point {i + 1} (curvature change={change:.1f}°)"
    return None


def _check_xseg_mismatch(landmarks, dflimg, image_size):
    """If XSeg mask exists, compare its area with landmark hull area."""
    if not dflimg.has_xseg_mask():
        return None

    try:
        xseg_mask = dflimg.get_xseg_mask()
        if xseg_mask is None:
            return None

        # Resize to image size if needed
        h_mask = xseg_mask.shape[0]
        if h_mask != image_size:
            xseg_mask = cv2.resize(xseg_mask, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if len(xseg_mask.shape) == 3:
            xseg_mask = xseg_mask[:, :, 0]

        xseg_area = float(np.sum(xseg_mask > 0.5))

        # Compute landmark convex hull area
        hull_mask = LandmarksProcessor.get_image_hull_mask(
            (image_size, image_size, 3), landmarks
        )
        hull_area = float(np.sum(hull_mask > 0.5))

        if hull_area < 1:
            return None

        ratio = xseg_area / hull_area
        if ratio < XSEG_HULL_MISMATCH:
            return f"XSeg/hull mismatch (ratio={ratio:.2f}, expected>{XSEG_HULL_MISMATCH})"
    except Exception:
        pass  # don't fail the scan if XSeg check has issues

    return None


def _check_duplicate_points(landmarks):
    """Check if any non-adjacent landmark points overlap (< 1px apart)."""
    # Define which point indices are naturally adjacent in the 68-point model
    # and thus allowed to be close together
    n = len(landmarks)
    bad_pairs = []
    for i in range(n):
        for j in range(i + 2, n):  # skip immediate neighbor
            dist = npla.norm(landmarks[i] - landmarks[j])
            if dist < DUPLICATE_DIST_MIN:
                bad_pairs.append((i, j))
                if len(bad_pairs) >= 3:
                    break
        if len(bad_pairs) >= 3:
            break

    if bad_pairs:
        pairs_str = ", ".join([f"{a}-{b}" for a, b in bad_pairs[:3]])
        return f"Duplicate/overlapping points at indices {pairs_str}"
    return None


def _check_nose_bridge(landmarks, abs_yaw):
    """Check if nose bridge points 27-30 form a roughly vertical line."""
    if abs_yaw > YAW_MILD:
        return None  # nose bridge naturally shifts for turned faces

    nose_bridge = landmarks[27:31]  # points 27, 28, 29, 30

    # Measure horizontal displacement relative to the eye distance
    r_eye_c = landmarks[36:42].mean(axis=0)
    l_eye_c = landmarks[42:48].mean(axis=0)
    eye_dist = npla.norm(l_eye_c - r_eye_c)
    if eye_dist < 1e-6:
        return None

    # Check if any bridge point deviates excessively from the vertical
    # line through the midpoint of the eyes
    mid_x = (r_eye_c[0] + l_eye_c[0]) / 2.0
    max_deviation = 0.0
    for pt in nose_bridge:
        deviation = abs(pt[0] - mid_x) / eye_dist
        max_deviation = max(max_deviation, deviation)

    if max_deviation > NOSE_BRIDGE_DEVIATION:
        return f"Nose bridge zigzag (deviation={max_deviation:.2f}, max={NOSE_BRIDGE_DEVIATION})"
    return None


def _check_source_rect_size(dflimg):
    """Warn if the original face detection was very small (unreliable landmarks)."""
    rect = dflimg.get_source_rect()
    if rect is None:
        return None

    try:
        l, t, r, b = rect
        rect_w = r - l
        rect_h = b - t
        rect_size = min(rect_w, rect_h)
        if rect_size < SOURCE_RECT_MIN_PX:
            return f"Tiny source face ({rect_size:.0f}px, min={SOURCE_RECT_MIN_PX}px)"
    except Exception:
        pass
    return None


def _check_blur(filepath):
    """Flag images that are too blurry for reliable landmarks."""
    try:
        img = cv2_imread(str(filepath))
        if img is None:
            return None
        sharpness = estimate_sharpness(img)
        if sharpness < BLUR_THRESHOLD:
            return f"Blurry source (sharpness={sharpness:.1f}, min={BLUR_THRESHOLD})"
    except Exception:
        pass
    return None


# ─── Main scanner ─────────────────────────────────────────────────

def _estimate_abs_yaw(landmarks):
    """Estimate absolute yaw (how far face is turned) from landmarks."""
    try:
        yaw = LandmarksProcessor.estimate_averaged_yaw(landmarks)
        # Normalize: estimate_averaged_yaw returns a difference in pixels,
        # convert to rough angle. The value depends on face size, so normalize
        # by inter-eye distance.
        r_eye_c = landmarks[36:42].mean(axis=0)
        l_eye_c = landmarks[42:48].mean(axis=0)
        eye_dist = npla.norm(l_eye_c - r_eye_c)
        if eye_dist < 1e-6:
            return math.pi / 2  # degenerate → treat as extreme
        # Rough mapping: yaw / eye_dist → radians
        # When yaw_diff == eye_dist, face is ~45°
        ratio = abs(yaw) / eye_dist
        return min(ratio * math.radians(45), math.pi / 2)
    except Exception:
        return 0.0


def scan_landmarks(input_path):
    """Scan a faceset folder for faces with bad landmarks.

    Args:
        input_path: Path to the aligned faces directory.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        io.log_err(f"Input path not found: {input_path}")
        return

    # Check for packed faceset (.pak file)
    pak_files = list(input_path.glob("*.pak"))
    if pak_files:
        io.log_err("")
        io.log_err("=" * 60)
        io.log_err("  PACKED FACESET DETECTED (.pak file)")
        io.log_err("=" * 60)
        io.log_err(f"  Found: {pak_files[0].name}")
        io.log_err("")
        io.log_err("  The faceset must be unpacked before scanning landmarks.")
        io.log_err("  Please run the 'unpack faceset' bat file first:")
        io.log_err("    • 4.2) data_src util faceset unpack.bat  (for SRC)")
        io.log_err("    • 5.2) data_dst util faceset unpack.bat  (for DST)")
        io.log_err("=" * 60)
        return

    image_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    total = len(image_paths)

    if total == 0:
        io.log_info("No images found in the directory.")
        return

    io.log_info(f"Scanning {total} faces for bad landmarks...\n")

    bad_files = []   # list of (filepath, [reasons])
    issue_counts = {}

    for filepath in io.progress_bar_generator(image_paths, "Scanning"):
        try:
            dflimg = DFLIMG.load(filepath)
            if dflimg is None or not dflimg.has_data():
                continue

            landmarks = dflimg.get_landmarks()
            if landmarks is None or len(landmarks) != 68:
                bad_files.append((filepath, ["Missing or invalid landmarks (not 68 points)"]))
                issue_counts["invalid_count"] = issue_counts.get("invalid_count", 0) + 1
                continue

            shape = dflimg.get_shape()
            if shape is None:
                continue
            image_size = shape[1]  # width

            # Estimate yaw for yaw-aware checks
            abs_yaw = _estimate_abs_yaw(landmarks)

            # Run all checks
            reasons = []
            checks = [
                _check_nan_inf(landmarks),
                _check_out_of_bounds(landmarks, image_size),
                _check_collapsed(landmarks, image_size),
                _check_eye_aspect(landmarks, abs_yaw),
                _check_asymmetry(landmarks, abs_yaw),
                _check_jaw_self_intersection(landmarks, abs_yaw),
                _check_jaw_smoothness(landmarks, abs_yaw),
                _check_xseg_mismatch(landmarks, dflimg, image_size),
                _check_duplicate_points(landmarks),
                _check_nose_bridge(landmarks, abs_yaw),
                _check_source_rect_size(dflimg),
                _check_blur(filepath),
            ]

            for result in checks:
                if result is not None:
                    reasons.append(result)
                    # Track issue type
                    issue_type = result.split("(")[0].strip()
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            if reasons:
                bad_files.append((filepath, reasons))

        except Exception as e:
            io.log_info(f"  Error processing {filepath.name}: {e}")

    # ─── Report ───
    io.log_info("")
    io.log_info("=" * 60)
    io.log_info(f"  LANDMARK SCAN RESULTS")
    io.log_info("=" * 60)
    io.log_info(f"  Total faces scanned:  {total}")
    io.log_info(f"  Bad landmarks found:  {len(bad_files)}")
    io.log_info(f"  Clean faces:          {total - len(bad_files)}")
    io.log_info("")

    if issue_counts:
        io.log_info("  Issue breakdown:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            io.log_info(f"    • {issue}: {count}")
        io.log_info("")

    if len(bad_files) > 0:
        # Show first 20 flagged files
        show_count = min(len(bad_files), 20)
        io.log_info(f"  First {show_count} flagged files:")
        for filepath, reasons in bad_files[:show_count]:
            io.log_info(f"    {filepath.name}")
            for r in reasons:
                io.log_info(f"      → {r}")
        if len(bad_files) > show_count:
            io.log_info(f"    ... and {len(bad_files) - show_count} more")
        io.log_info("")

        io.log_info("=" * 60)
        answer = io.input_str(
            f"\nWould you like to re-extract landmarks for these {len(bad_files)} faces? (y/n)",
            "n",
            ["y", "n"]
        )

        if answer.lower() == "y":
            fix_landmarks(input_path, bad_files)
        else:
            io.log_info("No changes made.")
    else:
        io.log_info("  All landmarks look good! No fixes needed.")
        io.log_info("=" * 60)


# ─── Quality scoring ──────────────────────────────────────────────

def _count_issues(landmarks, image_size, abs_yaw=None):
    """Count total heuristic failures for a set of landmarks.

    Returns an integer score — lower is better.  Used to compare old vs
    new landmarks before committing a fix.
    """
    if abs_yaw is None:
        abs_yaw = _estimate_abs_yaw(landmarks)

    score = 0
    # Critical checks (weight 3)
    if _check_nan_inf(landmarks) is not None:
        score += 3
    if _check_out_of_bounds(landmarks, image_size) is not None:
        score += 3
    if _check_collapsed(landmarks, image_size) is not None:
        score += 3

    # Structural checks (weight 2)
    if _check_eye_aspect(landmarks, abs_yaw) is not None:
        score += 2
    if _check_jaw_self_intersection(landmarks, abs_yaw) is not None:
        score += 2
    if _check_jaw_smoothness(landmarks, abs_yaw) is not None:
        score += 2
    if _check_duplicate_points(landmarks) is not None:
        score += 2

    # Geometric checks (weight 1)
    if _check_asymmetry(landmarks, abs_yaw) is not None:
        score += 1
    if _check_nose_bridge(landmarks, abs_yaw) is not None:
        score += 1

    return score


# ─── Fixer ────────────────────────────────────────────────────────

def fix_landmarks(input_path, bad_files):
    """Re-extract landmarks for flagged faces using FAN with two-pass S3FD.

    Uses S3FD to detect a proper tight face bounding box on the aligned
    image, then passes that to FAN with two-pass refinement enabled.
    Only saves the new landmarks if they score strictly better than the
    old ones on a weighted heuristic quality check.

    Args:
        input_path: Path to aligned faces directory.
        bad_files: list of (filepath, [reasons]) tuples.
    """
    io.log_info(f"\nRe-extracting landmarks for {len(bad_files)} faces...")
    io.log_info("Initializing face detector (S3FD) and landmark extractor (FAN)...\n")

    from core.leras import nn
    from facelib import S3FDExtractor, FANExtractor

    # Initialize on GPU
    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)

    s3fd = S3FDExtractor()
    fan = FANExtractor()

    fixed = 0
    failed = 0
    skipped_no_improve = 0

    for filepath, reasons in io.progress_bar_generator(bad_files, "Fixing landmarks"):
        try:
            dflimg = DFLIMG.load(filepath)
            if dflimg is None or not dflimg.has_data():
                failed += 1
                continue

            img = cv2_imread(str(filepath))
            if img is None:
                failed += 1
                continue

            h, w = img.shape[:2]

            # ── Step 1: Get a proper tight face rect via S3FD ──
            rects = s3fd.extract(img, is_bgr=True)

            if rects:
                # Use only the best (first / highest-confidence) detection
                rects = rects[0:1]
            else:
                # Fallback: synthetic rect from existing landmark bounding box
                old_lmrks = dflimg.get_landmarks()
                if old_lmrks is not None and len(old_lmrks) == 68:
                    lm = np.array(old_lmrks, dtype=np.float32)
                    mn = lm.min(axis=0)
                    mx = lm.max(axis=0)
                    pad_w = (mx[0] - mn[0]) * 0.20
                    pad_h = (mx[1] - mn[1]) * 0.20
                    rects = [(
                        max(0, int(mn[0] - pad_w)),
                        max(0, int(mn[1] - pad_h)),
                        min(w, int(mx[0] + pad_w)),
                        min(h, int(mx[1] + pad_h)),
                    )]
                else:
                    io.log_info(f"  {filepath.name}: S3FD found no face & no fallback, skipping")
                    failed += 1
                    continue

            # ── Step 2: Extract landmarks with FAN + two-pass S3FD refinement ──
            landmarks_list = fan.extract(
                img,
                rects,
                second_pass_extractor=s3fd,
                is_bgr=True,
                multi_sample=True
            )

            if not landmarks_list or landmarks_list[0] is None:
                io.log_info(f"  {filepath.name}: FAN returned no landmarks, skipping")
                failed += 1
                continue

            new_landmarks = landmarks_list[0]

            # ── Step 3: Quality gate — only save if strictly better ──
            old_landmarks = dflimg.get_landmarks()
            old_score = 999  # assume worst if missing
            if old_landmarks is not None and len(old_landmarks) == 68:
                old_landmarks = np.array(old_landmarks, dtype=np.float32)
                old_score = _count_issues(old_landmarks, w)

            new_score = _count_issues(new_landmarks, w)

            if new_score >= old_score:
                # New landmarks are not better — don't overwrite
                skipped_no_improve += 1
                continue

            # ── Step 4: Save improved landmarks (aligned-space only) ──
            # NOTE: We do NOT update source_landmarks — the merger uses those
            # (original-frame-space) and we only have the aligned crop.
            dflimg.set_landmarks(new_landmarks.tolist())
            dflimg.save()
            fixed += 1

        except Exception as e:
            io.log_info(f"  Error fixing {filepath.name}: {e}")
            failed += 1

    io.log_info("")
    io.log_info("=" * 60)
    io.log_info(f"  LANDMARK FIX RESULTS")
    io.log_info("=" * 60)
    io.log_info(f"  Successfully fixed:  {fixed}/{len(bad_files)}")
    if skipped_no_improve > 0:
        io.log_info(f"  Skipped (no improve): {skipped_no_improve}")
    if failed > 0:
        io.log_info(f"  Failed/skipped:      {failed}")
    io.log_info("")
    io.log_info("  Tip: Run '5.2) data_dst util add landmarks debug images.bat'")
    io.log_info("  (or the SRC equivalent) to visually verify the new landmarks.")
    io.log_info("=" * 60)
