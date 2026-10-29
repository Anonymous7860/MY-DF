import cv2
import numpy as np


OUTER_MOUTH_SLICE = slice(48, 60)
INNER_MOUTH_SLICE = slice(60, 68)


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n


def get_teeth_crop_affine(landmarks, output_size):
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.ndim != 2 or landmarks.shape[0] < 68 or landmarks.shape[1] != 2:
        raise ValueError("Expected 68 facial landmarks for teeth crop.")

    outer = landmarks[OUTER_MOUTH_SLICE]
    inner = landmarks[INNER_MOUTH_SLICE]
    pts = np.concatenate([outer, inner], axis=0)

    left_corner = outer[0]
    right_corner = outer[6]
    x_axis = _normalize(right_corner - left_corner)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

    upper_center = np.mean(np.concatenate([outer[2:5], inner[1:4]], axis=0), axis=0)
    lower_center = np.mean(np.concatenate([outer[8:11], inner[5:8]], axis=0), axis=0)
    if np.dot(lower_center - upper_center, y_axis) < 0:
        y_axis = -y_axis

    center = np.mean(pts, axis=0)
    rel = pts - center[None, :]
    x_proj = rel @ x_axis
    y_proj = rel @ y_axis

    x0 = float(np.min(x_proj))
    x1 = float(np.max(x_proj))
    y0 = float(np.min(y_proj))
    y1 = float(np.max(y_proj))

    mouth_w = max(4.0, x1 - x0)
    mouth_h = max(4.0, y1 - y0)

    # Teeth-only XSeg should spend almost all of its capacity on the mouth.
    # Use mouth-landmark bounds plus asymmetric padding that preserves lips,
    # lower teeth, and chin context while cutting off ears, temples, and brow.
    pad_x = mouth_w * 0.20
    pad_top = mouth_h * 0.42
    pad_bottom = mouth_h * 0.62

    crop_x0 = x0 - pad_x
    crop_x1 = x1 + pad_x
    crop_y0 = y0 - pad_top
    crop_y1 = y1 + pad_bottom

    side = max(crop_x1 - crop_x0, crop_y1 - crop_y0, 20.0)
    cx = (crop_x0 + crop_x1) * 0.5
    cy = (crop_y0 + crop_y1) * 0.5
    half = side * 0.5

    src = np.float32([
        center + x_axis * (cx - half) + y_axis * (cy - half),
        center + x_axis * (cx + half) + y_axis * (cy - half),
        center + x_axis * (cx - half) + y_axis * (cy + half),
    ])
    dst = np.float32([
        [0, 0],
        [output_size - 1, 0],
        [0, output_size - 1],
    ])

    mat = cv2.getAffineTransform(src, dst)
    inv_mat = cv2.getAffineTransform(dst, src)
    return mat, inv_mat


def warp_teeth_crop(image, landmarks, output_size, border_replicate=True, is_mask=False):
    mat, inv_mat = get_teeth_crop_affine(landmarks, output_size)
    border_mode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
    interp = cv2.INTER_LINEAR if is_mask else cv2.INTER_LANCZOS4
    cropped = cv2.warpAffine(
        image,
        mat,
        (output_size, output_size),
        flags=interp,
        borderMode=border_mode,
        borderValue=0.0,
    )
    if len(image.shape) == 3 and len(cropped.shape) == 2:
        cropped = cropped[..., None]
    return cropped, mat, inv_mat


def restore_teeth_crop(mask_crop, inv_mat, output_shape):
    h, w = output_shape[:2]
    restored = cv2.warpAffine(
        mask_crop,
        inv_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    if len(mask_crop.shape) == 3 and len(restored.shape) == 2:
        restored = restored[..., None]
    return restored
