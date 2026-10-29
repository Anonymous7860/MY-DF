import copy
import shutil
from pathlib import Path

import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from DFLIMG import DFLIMG, DFLJPG
from facelib import LandmarksProcessor


def _normalize_mask(mask, image_shape):
    mask = imagelib.normalize_channels(mask, 1).astype(np.float32)

    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(mask.shape) == 2:
            mask = mask[..., None]

    return np.clip(mask, 0.0, 1.0)


def _get_face_mask(dflimg, image_shape):
    xseg_mask = dflimg.get_xseg_mask()
    if xseg_mask is not None:
        xseg_mask = _normalize_mask(xseg_mask, image_shape)
        if float(xseg_mask.max()) > 0.01:
            return xseg_mask, "xseg"

    landmarks = dflimg.get_landmarks()
    eyebrows_expand_mod = dflimg.get_eyebrows_expand_mod()
    hull_mask = LandmarksProcessor.get_image_hull_mask(image_shape, landmarks, eyebrows_expand_mod)
    hull_mask = _normalize_mask(hull_mask, image_shape)

    blur = max(3, (min(image_shape[0], image_shape[1]) // 64) | 1)
    hull_mask = cv2.GaussianBlur(hull_mask, (blur, blur), 0)
    hull_mask = _normalize_mask(hull_mask, image_shape)

    return hull_mask, "hull"


def _get_edge_band(mask):
    mask_2d = mask[..., 0]
    h, w = mask_2d.shape
    edge_px = max(2, min(h, w) // 96)
    kernel_size = edge_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask_2d, kernel, iterations=1)
    band = np.clip(mask_2d - eroded, 0.0, 1.0)

    if float(band.sum()) < 1e-4:
        band = mask_2d

    return band


def _choose_auto_background(image_paths):
    if len(image_paths) <= 128:
        sample_paths = image_paths
    else:
        sample_indices = np.linspace(0, len(image_paths) - 1, num=128, dtype=np.int32)
        sample_paths = [image_paths[idx] for idx in sample_indices]

    edge_luma_sum = 0.0
    edge_weight_sum = 0.0

    for filepath in io.progress_bar_generator(sample_paths, "Sampling background color"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            continue

        img = cv2_imread(filepath)
        if img is None:
            continue

        mask, _ = _get_face_mask(dflimg, img.shape)
        band = _get_edge_band(mask)
        lab_l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0

        edge_luma_sum += float((lab_l * band).sum())
        edge_weight_sum += float(band.sum())

    if edge_weight_sum <= 1e-6:
        io.log_info("Auto mode fallback: insufficient mask edge signal, defaulting to black.")
        return "black", 0.5

    mean_edge_luma = edge_luma_sum / edge_weight_sum
    color_mode = "black" if mean_edge_luma >= 0.5 else "white"
    return color_mode, mean_edge_luma


def _apply_solid_background(img, mask, color_mode):
    bg_value = 0 if color_mode == "black" else 255
    bg = np.full_like(img, bg_value, dtype=np.uint8)

    alpha = mask.astype(np.float32)
    result = img.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _write_dfl_jpg(output_filepath, img, dfl_dict):
    cv2_imwrite(str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    output_dflimg = DFLJPG.load(output_filepath)
    if output_dflimg is None:
        raise Exception(f"Unable to reload {output_filepath.name} after writing")

    output_dflimg.set_dict(copy.deepcopy(dfl_dict))
    output_dflimg.save()


def process_folder(dirpath, background_mode="auto"):
    dirpath = Path(dirpath)
    image_paths = [Path(x) for x in pathex.get_image_paths(dirpath, image_extensions=[".jpg"], return_Path_class=False)]

    if len(image_paths) == 0:
        io.log_err(f"No JPG aligned faces found in {dirpath}")
        return

    if background_mode == "auto":
        background_mode, mean_edge_luma = _choose_auto_background(image_paths)
        io.log_info(
            f"Auto-selected {background_mode} background "
            f"(average face-edge lightness: {mean_edge_luma:.3f})"
        )
    else:
        mean_edge_luma = None
        io.log_info(f"Using forced {background_mode} background.")

    output_dirpath = dirpath.parent / (dirpath.name + "_solidbg")
    output_dirpath.mkdir(exist_ok=True, parents=True)

    output_images_paths = pathex.get_image_paths(output_dirpath)
    if len(output_images_paths) > 0:
        for filename in output_images_paths:
            Path(filename).unlink()

    dirpath_parts = "/".join(dirpath.parts[-2:])
    output_dirpath_parts = "/".join(output_dirpath.parts[-2:])
    io.log_info(f"Creating solid-background faceset from {dirpath_parts}")
    io.log_info(f"Processing to {output_dirpath_parts}")

    processed = []
    xseg_count = 0
    hull_count = 0

    for filepath in io.progress_bar_generator(image_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_err(f"{filepath.name} is not a DFL image file")
            continue

        img = cv2_imread(filepath)
        if img is None:
            io.log_err(f"Unable to read {filepath.name}")
            continue

        mask, mask_source = _get_face_mask(dflimg, img.shape)
        if mask_source == "xseg":
            xseg_count += 1
        else:
            hull_count += 1

        out_img = _apply_solid_background(img, mask, background_mode)
        output_filepath = output_dirpath / filepath.name
        _write_dfl_jpg(output_filepath, out_img, dflimg.get_dict())
        processed.append((filepath, output_filepath))

    io.log_info(
        f"Finished solid background pass: {len(processed)} files "
        f"({xseg_count} XSeg masks, {hull_count} landmark hull masks)."
    )

    is_merge = io.input_bool(f"\r\nMerge {output_dirpath_parts} to {dirpath_parts} ?", True)
    if is_merge:
        io.log_info(f"Copying processed files to {dirpath_parts}")
        for filepath, output_filepath in processed:
            shutil.copy(output_filepath, filepath)

        io.log_info(f"Removing {output_dirpath_parts}")
        shutil.rmtree(output_dirpath)
