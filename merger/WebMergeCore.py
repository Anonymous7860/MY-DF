import cv2
import numpy as np


def apply_clahe(image, amount):
    amount = float(np.clip(amount, 0, 100)) / 100.0
    if amount <= 0.0:
        return image
    src = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    lab = cv2.cvtColor(np.rint(src * 255.0).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + amount * 3.0, tileGridSize=(8, 8))
    corrected = cv2.cvtColor(
        cv2.merge((clahe.apply(l_channel), a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    ).astype(np.float32) / 255.0
    return np.clip(src * (1.0 - amount) + corrected * amount, 0.0, 1.0)


def match_color_statistics(image, reference, mask=None):
    result = np.asarray(image, dtype=np.float32).copy()
    reference = np.asarray(reference, dtype=np.float32)
    if mask is None:
        weights = np.ones(result.shape[:2], dtype=np.float32)
    else:
        weights = np.clip(np.asarray(mask, dtype=np.float32).squeeze(), 0.0, 1.0)
    valid = weights > (1.0 / 255.0)
    if not np.any(valid):
        return result
    weights_valid = weights[valid]
    weight_sum = float(weights_valid.sum()) + 1e-6
    for channel in range(3):
        src = result[..., channel][valid]
        ref = reference[..., channel][valid]
        src_mean = float(np.sum(src * weights_valid) / weight_sum)
        ref_mean = float(np.sum(ref * weights_valid) / weight_sum)
        src_std = float(np.sqrt(np.sum(((src - src_mean) ** 2) * weights_valid) / weight_sum)) + 1e-6
        ref_std = float(np.sqrt(np.sum(((ref - ref_mean) ** 2) * weights_valid) / weight_sum)) + 1e-6
        result[..., channel] = (result[..., channel] - src_mean) * (ref_std / src_std) + ref_mean
    return np.clip(result, 0.0, 1.0)


def blend_two_pass(first, second, mode, strength):
    first_face, first_swap_mask, first_dst_mask = first
    second_face, second_swap_mask, second_dst_mask = second
    alpha = float(np.clip(strength, 0, 100)) / 100.0
    second_face = match_color_statistics(second_face, first_face, first_swap_mask)
    face = np.clip(first_face * (1.0 - alpha) + second_face * alpha, 0.0, 1.0)

    swap_mask = first_swap_mask
    dst_mask = first_dst_mask
    if mode == "face+mask":
        swap_refined = np.clip(first_swap_mask * second_swap_mask, 0.0, 1.0)
        dst_refined = np.clip(first_dst_mask * second_dst_mask, 0.0, 1.0)
        swap_mask = np.clip(first_swap_mask * (1.0 - alpha) + swap_refined * alpha, 0.0, 1.0)
        dst_mask = np.clip(first_dst_mask * (1.0 - alpha) + dst_refined * alpha, 0.0, 1.0)
    return face, swap_mask, dst_mask


def clean_matte(mask, strength):
    strength = int(np.clip(strength, 0, 100))
    if strength <= 0:
        return np.clip(mask, 0.0, 1.0)
    radius = max(1, int(round(1 + 4 * strength / 100.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    work = np.clip(mask, 0.0, 1.0).astype(np.float32)
    opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    alpha = strength / 100.0
    return np.clip(work * (1.0 - alpha) + closed * alpha, 0.0, 1.0)


def apply_inward_feather(mask, amount, reference_dim=256):
    amount = int(max(0, amount))
    work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if amount <= 0 or not np.any(work > (1.0 / 255.0)):
        return work
    scale = min(work.shape[:2]) / float(max(16, reference_dim))
    width = max(0.75, amount * max(0.5, np.sqrt(scale)) / 2.0)
    hard = (work > (1.0 / 255.0)).astype(np.uint8)
    distance = cv2.distanceTransform(hard, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    t = np.clip(distance / width, 0.0, 1.0)
    ramp = 1.0 - ((1.0 - t) ** 3)
    return np.clip(np.minimum(work, ramp), 0.0, 1.0).astype(np.float32)


def apply_containment(mask, support, strength, feather=0):
    strength = float(np.clip(strength, 0, 100)) / 100.0
    if strength <= 0.0 or support is None:
        return np.clip(mask, 0.0, 1.0)
    support = np.clip(np.asarray(support, dtype=np.float32), 0.0, 1.0)
    if support.shape != mask.shape:
        support = cv2.resize(support, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
    if feather > 0:
        sigma = max(0.5, feather / 12.0)
        support = cv2.GaussianBlur(support, (0, 0), sigma)
    contained = np.minimum(np.clip(mask, 0.0, 1.0), support)
    return np.clip(mask * (1.0 - strength) + contained * strength, 0.0, 1.0)


def build_occlusion_keep_mask(dst_face, base_mask, strength):
    strength = int(np.clip(strength, 0, 100))
    base = np.clip(np.asarray(base_mask, dtype=np.float32), 0.0, 1.0)
    if strength <= 0 or not np.any(base > (1.0 / 255.0)):
        return np.ones_like(base, dtype=np.float32)

    gray = cv2.cvtColor(np.clip(dst_face * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    edge_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(edge_x, edge_y)
    edges = cv2.GaussianBlur(edges, (0, 0), 0.8 + 2.8 * strength / 100.0)
    edge_peak = float(edges.max())
    if edge_peak <= 1e-6:
        return np.ones_like(base, dtype=np.float32)
    edges /= edge_peak

    boundary_x = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    boundary_y = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    boundary = cv2.magnitude(boundary_x, boundary_y)
    boundary = cv2.GaussianBlur(boundary, (0, 0), 1.0 + 2.0 * strength / 100.0)
    boundary_peak = float(boundary.max())
    if boundary_peak <= 1e-6:
        return np.ones_like(base, dtype=np.float32)
    boundary /= boundary_peak

    score = np.sqrt(np.clip(edges * boundary, 0.0, 1.0))
    threshold = 0.48 - 0.24 * strength / 100.0
    activated = np.clip((score - threshold) / max(1e-4, 1.0 - threshold), 0.0, 1.0)
    radius = max(1, int(round(1 + strength / 45.0)))
    activated = cv2.dilate(
        activated,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2),
    )
    spread = cv2.GaussianBlur(activated, (0, 0), 0.7 + 1.5 * strength / 100.0)
    suppression = 0.18 + 0.82 * strength / 100.0
    return np.clip(1.0 - spread * boundary * suppression, 0.0, 1.0).astype(np.float32)


def _seam_weights(mask, amount):
    alpha = np.clip(mask, 0.0, 1.0).astype(np.float32)
    short_side = min(alpha.shape[:2])
    hard = (alpha > (1.0 / 255.0)).astype(np.uint8)
    distance = cv2.distanceTransform(hard, cv2.DIST_L2, 3)

    def band(minimum, span):
        width = max(2.0, short_side * (minimum + amount * span))
        weight = np.clip(1.0 - distance / width, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        return weight * alpha

    return band(0.04, 0.50)[..., None], band(0.03, 0.11)[..., None]


def _low_pass(image):
    height, width = image.shape[:2]
    short_side = min(height, width)
    scale = max(1, int(round(short_side / 24.0)))
    if scale > 1 and min(height, width) >= scale * 2:
        small = cv2.resize(image, (max(1, width // scale), max(1, height // scale)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.0)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(image, (0, 0), max(1.0, short_side / 24.0))


def apply_color_membrane(face, dst_face, mask, amount, luma_only=False):
    amount = float(np.clip(amount, 0, 100)) / 100.0
    if amount <= 0.0:
        return np.clip(face, 0.0, 1.0)
    tone_weight, detail_weight = _seam_weights(mask, amount)
    source = np.clip(face, 0.0, 1.0).astype(np.float32)
    destination = np.clip(dst_face, 0.0, 1.0).astype(np.float32)

    if luma_only:
        src_ycc = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
        dst_ycc = cv2.cvtColor(destination, cv2.COLOR_BGR2YCrCb)
        diff = np.zeros_like(src_ycc)
        diff[..., 0] = dst_ycc[..., 0] - src_ycc[..., 0]
        low = _low_pass(diff)
        correction = low * tone_weight + (diff - low) * detail_weight
        src_ycc += correction
        return np.clip(cv2.cvtColor(src_ycc, cv2.COLOR_YCrCb2BGR), 0.0, 1.0)

    diff = destination - source
    low = _low_pass(diff)
    correction = low * tone_weight + (diff - low) * detail_weight
    return np.clip(source + correction, 0.0, 1.0)


def apply_edge_repair(face, dst_face, mask, edge_clean, edge_decontam):
    edge_clean = float(np.clip(edge_clean, 0, 100)) / 100.0
    edge_decontam = float(np.clip(edge_decontam, 0, 100)) / 100.0
    if edge_clean <= 0.0 and edge_decontam <= 0.0:
        return np.clip(face, 0.0, 1.0)

    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    edge = np.clip(mask - eroded, 0.0, 1.0)[..., None]
    result = np.clip(face, 0.0, 1.0).astype(np.float32).copy()
    destination = np.clip(dst_face, 0.0, 1.0).astype(np.float32)

    if edge_clean > 0.0:
        face_ycc = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
        dst_ycc = cv2.cvtColor(destination, cv2.COLOR_BGR2YCrCb)
        face_ycc[..., 0] += (dst_ycc[..., 0] - face_ycc[..., 0]) * edge[..., 0] * min(0.8, 0.28 + edge_clean * 0.2)
        face_ycc[..., 1:] += (dst_ycc[..., 1:] - face_ycc[..., 1:]) * edge * min(0.55, 0.16 + edge_clean * 0.16)
        result = cv2.cvtColor(np.clip(face_ycc, 0.0, 1.0), cv2.COLOR_YCrCb2BGR)

    if edge_decontam > 0.0:
        dst_soft = cv2.GaussianBlur(destination, (0, 0), 1.2 + edge_decontam * 2.4)
        target = result * 0.2 + dst_soft * 0.8
        weight = np.clip(edge * min(0.55, edge_decontam * 0.55), 0.0, 1.0)
        result = result * (1.0 - weight) + target * weight
    return np.clip(result, 0.0, 1.0)


def apply_detail(face, amount):
    amount = float(np.clip(amount, 0, 100)) / 100.0
    if amount <= 0.0:
        return np.clip(face, 0.0, 1.0)
    source = np.clip(face, 0.0, 1.0).astype(np.float32)
    low = cv2.GaussianBlur(source, (0, 0), 1.25)
    detail = source - low
    return np.clip(source + detail * (0.75 * amount), 0.0, 1.0)


def apply_finishing(face, config):
    result = np.clip(face, 0.0, 1.0).astype(np.float32)
    contrast = max(0.0, 1.0 + float(getattr(config, "contrast", 0)) / 100.0)
    brightness = float(getattr(config, "brightness", 0)) / 100.0 * 0.35
    result = (result - 0.5) * contrast + 0.5 + brightness

    saturation = max(0.0, 1.0 + float(getattr(config, "saturation", 0)) / 100.0)
    if abs(saturation - 1.0) > 1e-6:
        hsv = cv2.cvtColor(np.clip(result, 0.0, 1.0), cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    gamma_control = float(getattr(config, "gamma", 0))
    if abs(gamma_control) > 1e-6:
        gamma = 2.5 ** (gamma_control / 100.0)
        result = np.power(np.clip(result, 0.0, 1.0), 1.0 / max(0.1, gamma))

    offsets = np.array(
        [
            getattr(config, "blue_offset", 0),
            getattr(config, "green_offset", 0),
            getattr(config, "red_offset", 0),
        ],
        dtype=np.float32,
    ) / 100.0 * 0.25
    result += offsets.reshape(1, 1, 3)
    return np.clip(result, 0.0, 1.0)


def restore_local_features(face, dst_face, aligned_landmarks, config):
    if aligned_landmarks is None or len(aligned_landmarks) < 68:
        return face
    result = np.clip(face, 0.0, 1.0).astype(np.float32).copy()
    points = np.asarray(aligned_landmarks, dtype=np.float32)

    def restore(indices, power, grow, feather):
        nonlocal result
        power = float(np.clip(power, 0, 100)) / 100.0
        if power <= 0.0:
            return
        feature = points[indices].astype(np.int32)
        feature_mask = np.zeros(result.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(feature_mask, cv2.convexHull(feature), 255)
        radius = int(round(grow))
        if radius != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(radius) * 2 + 1,) * 2)
            feature_mask = cv2.dilate(feature_mask, kernel) if radius > 0 else cv2.erode(feature_mask, kernel)
        if feather > 0:
            sigma = max(0.5, float(feather) / 3.0)
            feature_mask = cv2.GaussianBlur(feature_mask, (0, 0), sigma)
        alpha = (feature_mask.astype(np.float32) / 255.0 * power)[..., None]
        result = result * (1.0 - alpha) + dst_face * alpha

    restore(list(range(36, 48)), getattr(config, "restore_eyes_power", 0), getattr(config, "restore_eyes_size", 12), getattr(config, "restore_eyes_blur", 10))
    restore(list(range(48, 68)), getattr(config, "restore_mouth_power", 0), getattr(config, "restore_mouth_size", 14), getattr(config, "restore_mouth_blur", 12))
    return np.clip(result, 0.0, 1.0)

