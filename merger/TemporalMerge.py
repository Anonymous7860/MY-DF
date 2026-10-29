from pathlib import Path

import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import cv2_imread, cv2_imwrite
from core.interact import interact as io


class TemporalMergeState:
    """Motion-compensated temporal fusion for already-merged masked frames.

    Only the low-frequency merge residual and alpha mask are accumulated. The
    current frame's high-frequency residual is always retained, preventing the
    temporal pass from smearing pores, beard, teeth, lashes, or expression
    changes. Flow confidence and photometric agreement reset unstable regions.
    """

    def __init__(self, strength=0.65):
        self.strength = float(np.clip(strength, 0.0, 0.95))
        self.reset()

    def reset(self):
        self.previous_original = None
        self.previous_mask = None
        self.previous_low_residual = None
        self.previous_index = None

    @staticmethod
    def _as_float_image(image, channels):
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[..., None]
        image = imagelib.normalize_channels(image, channels)
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return np.clip(image.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _low_frequency(image):
        height, width = image.shape[:2]
        sigma = float(np.clip(min(height, width) / 80.0, 2.0, 8.0))
        return cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT101,
        )

    @staticmethod
    def _mask_bounds(current_mask, previous_mask):
        support = (current_mask > (1.0 / 255.0)) | (previous_mask > (1.0 / 255.0))
        points = np.argwhere(support)
        if points.size == 0:
            return None

        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        height, width = current_mask.shape
        margin = max(16, int(round(max(y1 - y0, x1 - x0) * 0.20)))
        return (
            max(0, int(y0) - margin),
            min(height, int(y1) + margin),
            max(0, int(x0) - margin),
            min(width, int(x1) + margin),
        )

    @staticmethod
    def _flow_maps(current_gray, previous_gray):
        # Backward flow gives the previous-frame sample coordinate for each
        # current-frame pixel. Forward flow is used only for confidence.
        backward = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        forward = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        height, width = current_gray.shape
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        map_x = xx + backward[..., 0]
        map_y = yy + backward[..., 1]
        warped_forward_x = cv2.remap(
            forward[..., 0], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_forward_y = cv2.remap(
            forward[..., 1], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        fb_error = np.sqrt(
            np.square(backward[..., 0] + warped_forward_x)
            + np.square(backward[..., 1] + warped_forward_y)
        )
        inside = (
            (map_x >= 0.0)
            & (map_x <= width - 1.0)
            & (map_y >= 0.0)
            & (map_y <= height - 1.0)
        )
        return map_x, map_y, backward, fb_error, inside

    def _remember(self, frame_index, original, merged, mask, low_residual=None):
        residual = merged - original
        if low_residual is None:
            low_residual = self._low_frequency(residual)
        self.previous_original = original.copy()
        self.previous_mask = mask.copy()
        self.previous_low_residual = low_residual.copy()
        self.previous_index = int(frame_index)

    def process(self, frame_index, original_bgr, merged_bgr, merged_mask):
        original = self._as_float_image(original_bgr, 3)
        merged = self._as_float_image(merged_bgr, 3)
        mask = self._as_float_image(merged_mask, 1)[..., 0]

        if original.shape != merged.shape or mask.shape != original.shape[:2]:
            raise ValueError(
                "Temporal merger requires matching original, merged, and mask dimensions."
            )

        has_face = bool(np.any(mask > (1.0 / 255.0)))
        is_sequential = (
            self.previous_index is not None
            and int(frame_index) == self.previous_index + 1
        )
        state_matches = (
            self.previous_original is not None
            and self.previous_original.shape == original.shape
            and self.previous_mask.shape == mask.shape
        )

        if not has_face:
            self.reset()
            return merged, mask

        if not is_sequential or not state_matches or self.strength <= 0.0:
            self._remember(frame_index, original, merged, mask)
            return merged, mask

        bounds = self._mask_bounds(mask, self.previous_mask)
        if bounds is None:
            self._remember(frame_index, original, merged, mask)
            return merged, mask

        y0, y1, x0, x1 = bounds
        current_original_roi = original[y0:y1, x0:x1]
        previous_original_roi = self.previous_original[y0:y1, x0:x1]
        current_mask_roi = mask[y0:y1, x0:x1]
        previous_mask_roi = self.previous_mask[y0:y1, x0:x1]

        current_gray = cv2.cvtColor(current_original_roi, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_original_roi, cv2.COLOR_BGR2GRAY)
        map_x, map_y, backward, fb_error, inside = self._flow_maps(
            current_gray, previous_gray
        )

        warped_previous_gray = cv2.remap(
            previous_gray, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        warped_previous_mask = cv2.remap(
            previous_mask_roi, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        previous_low_roi = self.previous_low_residual[y0:y1, x0:x1]
        warped_previous_low = cv2.remap(
            previous_low_roi, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        photometric_error = np.abs(warped_previous_gray - current_gray)
        motion = np.sqrt(
            np.square(backward[..., 0]) + np.square(backward[..., 1])
        )
        support = np.maximum(current_mask_roi, warped_previous_mask) > (1.0 / 255.0)
        max_motion = 0.20 * float(max(current_gray.shape))
        stable = (
            inside
            & support
            & (photometric_error < 0.10)
            & (fb_error < 1.5)
            & (motion < max_motion)
        )

        support_count = int(np.count_nonzero(support))
        stable_count = int(np.count_nonzero(stable))
        if support_count == 0 or stable_count < max(16, int(0.12 * support_count)):
            self._remember(frame_index, original, merged, mask)
            return merged, mask

        stable_u8 = stable.astype(np.uint8)
        stable_u8 = cv2.morphologyEx(
            stable_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        confidence = cv2.GaussianBlur(
            stable_u8.astype(np.float32), (0, 0), sigmaX=1.25, sigmaY=1.25
        )
        confidence *= np.clip(1.0 - photometric_error / 0.10, 0.0, 1.0)
        confidence *= np.clip(1.0 - fb_error / 1.5, 0.0, 1.0)
        alpha = np.clip(confidence * self.strength, 0.0, self.strength)

        current_residual = merged - original
        current_low = self._low_frequency(current_residual)
        current_high = current_residual - current_low

        current_low_roi = current_low[y0:y1, x0:x1]
        alpha_rgb = alpha[..., None]
        fused_low_roi = (
            current_low_roi * (1.0 - alpha_rgb)
            + warped_previous_low * alpha_rgb
        )

        fused_low = current_low.copy()
        fused_low[y0:y1, x0:x1] = fused_low_roi
        output = np.clip(original + current_high + fused_low, 0.0, 1.0)

        # The mask receives a slightly gentler history contribution than color,
        # reducing edge chatter without making fast silhouettes lag behind.
        mask_alpha = alpha * 0.80
        fused_mask_roi = (
            current_mask_roi * (1.0 - mask_alpha)
            + warped_previous_mask * mask_alpha
        )
        fused_mask = mask.copy()
        fused_mask[y0:y1, x0:x1] = np.clip(fused_mask_roi, 0.0, 1.0)

        self._remember(
            frame_index,
            original,
            output,
            fused_mask,
            low_residual=fused_low,
        )
        return output, fused_mask


def process_temporal_sequence(
    input_image_paths,
    raw_output_path,
    raw_mask_path,
    output_path,
    output_mask_path,
    strength=0.65,
):
    """Create an ordered temporal sequence from private classic-merge frames."""
    raw_output_path = Path(raw_output_path)
    raw_mask_path = Path(raw_mask_path)
    output_path = Path(output_path)
    output_mask_path = Path(output_mask_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_mask_path.mkdir(parents=True, exist_ok=True)

    for filename in pathex.get_image_paths(output_path):
        Path(filename).unlink()
    for filename in pathex.get_image_paths(output_mask_path):
        Path(filename).unlink()

    state = TemporalMergeState(strength=strength)
    written = 0
    ordered_inputs = [Path(path) for path in input_image_paths]
    for frame_index in io.progress_bar_generator(
        range(len(ordered_inputs)), "Temporal fusion"
    ):
        input_filepath = ordered_inputs[frame_index]
        raw_filepath = raw_output_path / f"{input_filepath.stem}.png"
        raw_mask_filepath = raw_mask_path / f"{input_filepath.stem}.png"
        if not raw_filepath.exists() or not raw_mask_filepath.exists():
            state.reset()
            continue

        original = cv2_imread(input_filepath, verbose=False)
        merged = cv2_imread(raw_filepath, verbose=False)
        mask = cv2_imread(raw_mask_filepath, verbose=False)
        if original is None or merged is None or mask is None:
            state.reset()
            continue

        original = imagelib.normalize_channels(original, 3)
        merged = imagelib.normalize_channels(merged, 3)
        mask = imagelib.normalize_channels(mask, 1)
        output, output_mask = state.process(
            frame_index, original, merged, mask
        )
        cv2_imwrite(
            output_path / f"{input_filepath.stem}.png",
            np.rint(output * 255.0).astype(np.uint8),
        )
        cv2_imwrite(
            output_mask_path / f"{input_filepath.stem}.png",
            np.rint(output_mask[..., None] * 255.0).astype(np.uint8),
        )
        written += 1

    io.log_info(
        f"Temporal merger wrote {written} frame(s) to {output_path}."
    )
    return written
