from pathlib import Path

import cv2
import numpy as np

from core import pathex
from core.cv2ex import cv2_imread
from core.interact import interact as io
from core.leras import nn
from DFLIMG import DFLJPG
from facelib import FaceType, LandmarksProcessor
from .Extractor import ExtractSubprocessor


def dev_test(input_dir):
    """Extract aligned FaceSynthetics faces and attach their segmentation masks."""
    image_size = 1024
    face_type = FaceType.WHOLE_FACE
    input_path = Path(input_dir)

    output_path = input_path.parent / f'{input_path.name}_out'
    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if output_images_paths:
            io.input(
                f"\n WARNING !!! \n {output_path} contains files! \n "
                "They will be deleted. \n Press enter to continue.\n"
            )
            for filename in output_images_paths:
                Path(filename).unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    data = []
    for filepath in io.progress_bar_generator(pathex.get_paths(input_path), "Processing"):
        if filepath.suffix != '.txt':
            continue

        image_filepath = filepath.parent / f'{filepath.name.split("_")[0]}.png'
        if not image_filepath.exists():
            print(f'{image_filepath} does not exist, skipping')
            continue

        landmarks = []
        for landmark_line in filepath.read_text().split('\n'):
            if not landmark_line:
                continue
            x, y = landmark_line.split(' ')
            landmarks.append((float(x), float(y)))

        landmarks = np.array(landmarks[:68], np.float32)
        rect = LandmarksProcessor.get_rect_from_landmarks(landmarks)
        data.append(
            ExtractSubprocessor.Data(
                filepath=image_filepath,
                rects=[rect],
                landmarks=[landmarks],
            )
        )

    if not data:
        return

    io.log_info("Performing 3rd pass...")
    ExtractSubprocessor(
        data,
        'final',
        image_size,
        95,
        face_type,
        final_output_path=output_path,
        device_config=nn.DeviceConfig.CPU(),
    ).run()

    for filename in io.progress_bar_generator(pathex.get_image_paths(output_path), "Processing"):
        filepath = Path(filename)
        dflimg = DFLJPG.load(filepath)
        src_filename = dflimg.get_source_filename()
        image_to_face_mat = dflimg.get_image_to_face_mat()

        seg_filepath = input_path / (Path(src_filename).stem + '_seg.png')
        if not seg_filepath.exists():
            raise ValueError(f'{seg_filepath} does not exist')

        seg = cv2_imread(seg_filepath)
        seg_inds = np.isin(seg, [1, 2, 3, 4, 5, 6, 9, 10, 11])
        seg[~seg_inds] = 0
        seg[seg_inds] = 1
        seg = seg.astype(np.float32)
        seg = cv2.warpAffine(
            seg,
            image_to_face_mat,
            (image_size, image_size),
            cv2.INTER_LANCZOS4,
        )
        dflimg.set_xseg_mask(seg)
        dflimg.save()
