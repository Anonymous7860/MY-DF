import cv2
import numpy as np

from .FaceType import FaceType
from .FANExtractor import FANExtractor
from .S3FDExtractor import S3FDExtractor
from . import LandmarksProcessor


class SAENXTExtractor:
    """Adaptive, batched face extraction strategy for video frame sequences."""

    def __init__(self, landmarks_3D=False, place_model_on_cpu=False, device_total_vram_gb=0):
        self.detector = S3FDExtractor(place_model_on_cpu=place_model_on_cpu)
        self.landmarks = FANExtractor(
            landmarks_3D=landmarks_3D,
            place_model_on_cpu=place_model_on_cpu,
        )
        self.cpu_only = self.detector.device.type != 'cuda'
        self.device_total_vram_gb = float(device_total_vram_gb or 0)

        if self.cpu_only:
            self.frame_batch_size = 1
            self.landmark_batch_size = 2
            self.fallback_size = 768
        elif self.device_total_vram_gb < 4:
            self.frame_batch_size = 1
            self.landmark_batch_size = 2
            self.fallback_size = 768
        elif self.device_total_vram_gb < 6:
            self.frame_batch_size = 2
            self.landmark_batch_size = 4
            self.fallback_size = 832
        elif self.device_total_vram_gb < 10:
            self.frame_batch_size = 4
            self.landmark_batch_size = 8
            self.fallback_size = 896
        elif self.device_total_vram_gb < 16:
            self.frame_batch_size = 6
            self.landmark_batch_size = 12
            self.fallback_size = 960
        else:
            self.frame_batch_size = 8
            self.landmark_batch_size = 16
            self.fallback_size = 1024

        self.previous_primary_rect = None

    @staticmethod
    def rotate_image(image, rotation):
        if rotation == 90:
            return image.swapaxes(0, 1)[:, ::-1, :]
        if rotation == 180:
            return image[::-1, ::-1, :]
        if rotation == 270:
            return image.swapaxes(0, 1)[::-1, :, :]
        return image

    @staticmethod
    def map_to_original(rect, landmarks, rotation, original_shape):
        h, w = original_shape[:2]
        l, t, r, b = rect

        if rotation == 90:
            mapped_rect = (t, h - r, b, h - l)
            mapped_landmarks = landmarks[:, ::-1].copy() if landmarks is not None else None
            if mapped_landmarks is not None:
                mapped_landmarks[:, 1] = h - mapped_landmarks[:, 1]
        elif rotation == 180:
            mapped_rect = (w - r, h - b, w - l, h - t)
            mapped_landmarks = landmarks.copy() if landmarks is not None else None
            if mapped_landmarks is not None:
                mapped_landmarks[:, 0] = w - mapped_landmarks[:, 0]
                mapped_landmarks[:, 1] = h - mapped_landmarks[:, 1]
        elif rotation == 270:
            mapped_rect = (w - b, l, w - t, r)
            mapped_landmarks = landmarks[:, ::-1].copy() if landmarks is not None else None
            if mapped_landmarks is not None:
                mapped_landmarks[:, 0] = w - mapped_landmarks[:, 0]
        else:
            mapped_rect = (l, t, r, b)
            mapped_landmarks = landmarks

        mapped_rect = tuple(int(value) for value in mapped_rect)
        return mapped_rect, mapped_landmarks

    @staticmethod
    def _rect_size(rect):
        return max(1.0, float(min(rect[2] - rect[0], rect[3] - rect[1])))

    def _temporal_reason(self, mapped_rect):
        previous = self.previous_primary_rect
        if previous is None:
            return None

        current_center = np.array([(mapped_rect[0] + mapped_rect[2]) * 0.5,
                                   (mapped_rect[1] + mapped_rect[3]) * 0.5])
        previous_center = np.array([(previous[0] + previous[2]) * 0.5,
                                    (previous[1] + previous[3]) * 0.5])
        current_size = self._rect_size(mapped_rect)
        previous_size = self._rect_size(previous)
        center_jump = np.linalg.norm(current_center - previous_center) / max(current_size, previous_size)
        scale_ratio = max(current_size, previous_size) / max(1.0, min(current_size, previous_size))

        if center_jump > 0.85 or scale_ratio > 1.9:
            return 'temporal_jump'
        return None

    @staticmethod
    def _quality(landmarks, metrics, rect, image_shape, detection_score):
        reasons = []
        hard_failures = 0
        if landmarks is None or np.asarray(landmarks).shape != (68, 2):
            return 1000.0, ['missing_landmarks'], 1

        landmarks = np.asarray(landmarks, dtype=np.float32)
        if not np.all(np.isfinite(landmarks)):
            return 1000.0, ['invalid_landmarks'], 1

        h, w = image_shape[:2]
        rect_width = max(1.0, float(rect[2] - rect[0]))
        rect_height = max(1.0, float(rect[3] - rect[1]))
        rect_size = max(1.0, min(rect_width, rect_height))
        bbox_min = landmarks.min(axis=0)
        bbox_max = landmarks.max(axis=0)
        bbox_width = float(bbox_max[0] - bbox_min[0])
        bbox_height = float(bbox_max[1] - bbox_min[1])
        width_ratio = bbox_width / rect_width
        height_ratio = bbox_height / rect_height

        if width_ratio < 0.25 or width_ratio > 1.65:
            reasons.append('landmark_width')
            hard_failures += 1
        if height_ratio < 0.30 or height_ratio > 1.75:
            reasons.append('landmark_height')
            hard_failures += 1

        right_eye = landmarks[36:42].mean(axis=0)
        left_eye = landmarks[42:48].mean(axis=0)
        eye_distance = float(np.linalg.norm(left_eye - right_eye))
        if eye_distance < rect_size * 0.08:
            reasons.append('collapsed_eyes')
            hard_failures += 1

        try:
            transform = LandmarksProcessor.get_transform_mat(landmarks, 256, FaceType.FULL)
            if not np.all(np.isfinite(transform)):
                raise ValueError('non-finite transform')
            aligned = LandmarksProcessor.transform_points(landmarks, transform)
            aligned_eye_distance = float(np.linalg.norm(
                aligned[42:48].mean(axis=0) - aligned[36:42].mean(axis=0)
            ))
            yaw_signal = abs(LandmarksProcessor.estimate_averaged_yaw(aligned)) / max(aligned_eye_distance, 1.0)
            if yaw_signal > 0.42:
                reasons.append('profile')
        except Exception:
            reasons.append('invalid_transform')
            hard_failures += 1

        if rect_size < 180:
            reasons.append('small_face')
        if detection_score < 0.65:
            reasons.append('weak_detection')

        border_margin = max(2.0, rect_size * 0.02)
        if (rect[0] <= border_margin or rect[1] <= border_margin or
                rect[2] >= w - border_margin or rect[3] >= h - border_margin):
            reasons.append('frame_border')

        if metrics is not None:
            if metrics.get('peak_z_median', 99.0) < 3.0:
                reasons.append('weak_heatmaps')
            if metrics.get('edge_fraction', 0.0) > 0.04:
                reasons.append('heatmap_edge')
                hard_failures += 1
            if metrics.get('sample_spread', 0.0) > 0.018:
                reasons.append('unstable_samples')
                hard_failures += 1

        score = hard_failures * 100.0
        score += max(0.0, 4.0 - float((metrics or {}).get('peak_z_median', 4.0))) * 2.0
        score += float((metrics or {}).get('edge_fraction', 0.0)) * 25.0
        score += float((metrics or {}).get('sample_spread', 0.0)) * 200.0
        score += max(0.0, 0.75 - detection_score) * 4.0
        return score, reasons, hard_failures

    def _detect_rotated(self, image, rotation, target_size):
        rotated = self.rotate_image(image, rotation)
        detections = self.detector.extract_saenxt(
            rotated,
            target_size=target_size,
            is_bgr=True,
        )
        return rotated, detections

    def _refinement_mode(self, reasons, hard_failures, rect):
        reason_set = set(reasons)
        immediate_reasons = {
            'missing_landmarks',
            'invalid_landmarks',
            'landmark_width',
            'landmark_height',
            'collapsed_eyes',
            'invalid_transform',
            'weak_detection',
            'weak_heatmaps',
            'heatmap_edge',
            'unstable_samples',
            'temporal_jump',
        }
        immediate = hard_failures > 0 or bool(reason_set & immediate_reasons)
        if 'small_face' in reason_set and self._rect_size(rect) < 110:
            immediate = True

        if immediate:
            return 'full'
        if 'profile' in reason_set:
            return 'light'
        return None

    def _adaptive_detections(self, images, max_faces_from_image):
        upright = self.detector.extract_batch_saenxt(
            images,
            target_size=640,
            is_bgr=True,
            max_batch_size=self.frame_batch_size,
        )
        entries = []

        for image, primary in zip(images, upright):
            rotation = 0
            rotated_image = image
            detections = primary
            reasons = []

            if len(detections) == 0:
                for candidate_rotation in (90, 270, 180):
                    candidate_image, candidate = self._detect_rotated(image, candidate_rotation, 640)
                    if len(candidate) != 0:
                        rotation = candidate_rotation
                        rotated_image = candidate_image
                        detections = candidate
                        reasons.append('rotated_detection')
                        break

            weak_or_small = (
                len(detections) == 0 or
                detections[0][4] < 0.68 or
                self._rect_size(detections[0]) < 150
            )
            if weak_or_small:
                high_res_found = False
                rotations = (rotation,) if len(detections) != 0 else (0, 90, 270, 180)
                for candidate_rotation in rotations:
                    try:
                        candidate_image, candidate = self._detect_rotated(
                            image,
                            candidate_rotation,
                            self.fallback_size,
                        )
                    except RuntimeError:
                        candidate = []
                    if len(candidate) != 0:
                        rotation = candidate_rotation
                        rotated_image = candidate_image
                        detections = candidate
                        high_res_found = True
                        break
                if high_res_found:
                    reasons.append('high_res_detection')

            if max_faces_from_image is not None and max_faces_from_image > 0:
                detections = detections[:max_faces_from_image]

            entries.append({
                'rotated_image': rotated_image,
                'rotation': rotation,
                'detections': detections,
                'reasons': reasons,
            })

        return entries

    def extract_batch(self, images, max_faces_from_image=0):
        entries = self._adaptive_detections(images, max_faces_from_image)
        fan_items = [
            (entry['rotated_image'], [detection[:4] for detection in entry['detections']])
            for entry in entries
        ]
        first_landmarks, first_metrics = self.landmarks.extract_batch_saenxt(
            fan_items,
            multi_sample=False,
            max_batch_size=self.landmark_batch_size,
        )

        candidate_records = []
        refinement_images = []
        refinement_records = []

        for image_idx, (image, entry) in enumerate(zip(images, entries)):
            records = []
            primary_mapped_rect = None
            for face_idx, detection in enumerate(entry['detections']):
                rect = detection[:4]
                detection_score = float(detection[4])
                landmarks = first_landmarks[image_idx][face_idx]
                metrics = first_metrics[image_idx][face_idx]
                score, reasons, hard_failures = self._quality(
                    landmarks,
                    metrics,
                    rect,
                    entry['rotated_image'].shape,
                    detection_score,
                )

                mapped_rect, _ = self.map_to_original(rect, landmarks, entry['rotation'], image.shape)
                if face_idx == 0:
                    primary_mapped_rect = mapped_rect
                    temporal_reason = self._temporal_reason(mapped_rect)
                    if temporal_reason is not None:
                        reasons.append(temporal_reason)

                record = {
                    'rect': rect,
                    'detection_score': detection_score,
                    'landmarks': landmarks,
                    'metrics': metrics,
                    'score': score,
                    'hard_failures': hard_failures,
                    'reasons': reasons,
                    'refined_mode': None,
                }
                records.append(record)

                refinement_mode = self._refinement_mode(reasons, hard_failures, rect)
                if landmarks is not None and refinement_mode is not None:
                    try:
                        face_mat = LandmarksProcessor.get_transform_mat(landmarks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(
                            entry['rotated_image'],
                            face_mat,
                            (256, 256),
                            cv2.INTER_CUBIC,
                        )
                        refinement_images.append(face_image)
                        refinement_records.append((image_idx, face_idx, face_mat, refinement_mode))
                    except Exception:
                        pass
            candidate_records.append(records)
            self.previous_primary_rect = primary_mapped_rect if len(records) == 1 else None

        if len(refinement_images) != 0:
            inner_detections = self.detector.extract_batch_saenxt(
                refinement_images,
                target_size=256,
                is_bgr=True,
                max_batch_size=self.frame_batch_size,
            )
            grouped_items = {'light': [], 'full': []}
            grouped_records = {'light': [], 'full': []}
            for face_image, detections, refinement_record in zip(
                    refinement_images, inner_detections, refinement_records):
                if len(detections) == 1:
                    mode = refinement_record[3]
                    grouped_items[mode].append((face_image, [detections[0][:4]]))
                    grouped_records[mode].append((refinement_record, float(detections[0][4])))

            for mode in ('light', 'full'):
                refinement_items = grouped_items[mode]
                valid_refinement_records = grouped_records[mode]
                if len(refinement_items) == 0:
                    continue

                refined_landmarks, refined_metrics = self.landmarks.extract_batch_saenxt(
                    refinement_items,
                    multi_sample=mode == 'full',
                    max_batch_size=self.landmark_batch_size,
                )
                for refinement_idx, (refinement_record, inner_score) in enumerate(valid_refinement_records):
                    image_idx, face_idx, face_mat, _ = refinement_record
                    face_landmarks = refined_landmarks[refinement_idx][0]
                    if face_landmarks is None:
                        continue
                    candidate = candidate_records[image_idx][face_idx]
                    refined = LandmarksProcessor.transform_points(face_landmarks, face_mat, True)
                    refined_metric = refined_metrics[refinement_idx][0]
                    refined_score, refined_reasons, refined_hard = self._quality(
                        refined,
                        refined_metric,
                        candidate['rect'],
                        entries[image_idx]['rotated_image'].shape,
                        max(candidate['detection_score'], inner_score),
                    )

                    use_refined = (
                        refined_hard < candidate['hard_failures'] or
                        (refined_hard == 0 and refined_score <= candidate['score'] + 1.0) or
                        (refined_hard == 0 and any(
                            reason in candidate['reasons']
                            for reason in ('profile', 'small_face', 'weak_detection', 'weak_heatmaps')
                        ))
                    )
                    if use_refined:
                        candidate['landmarks'] = refined
                        candidate['metrics'] = refined_metric
                        candidate['score'] = refined_score
                        candidate['hard_failures'] = refined_hard
                        candidate['reasons'] = refined_reasons
                        candidate['refined_mode'] = mode

        results = []
        for image, entry, records in zip(images, entries, candidate_records):
            mapped_rects = []
            mapped_landmarks = []
            rect_scores = []
            frame_reasons = list(entry['reasons'])
            used_refinement = False
            used_full_refinement = False
            used_light_refinement = False

            for record in records:
                mapped_rect, mapped_points = self.map_to_original(
                    record['rect'],
                    record['landmarks'],
                    entry['rotation'],
                    image.shape,
                )
                mapped_rects.append(mapped_rect)
                mapped_landmarks.append(mapped_points)
                rect_scores.append(record['detection_score'])
                frame_reasons.extend(record['reasons'])
                used_refinement = used_refinement or record['refined_mode'] is not None
                used_full_refinement = used_full_refinement or record['refined_mode'] == 'full'
                used_light_refinement = used_light_refinement or record['refined_mode'] == 'light'

            if len(mapped_rects) == 0:
                frame_reasons.append('no_face')
                self.previous_primary_rect = None
            elif len(mapped_rects) == 1:
                self.previous_primary_rect = mapped_rects[0]
            else:
                self.previous_primary_rect = None

            frame_reasons = sorted(set(frame_reasons))
            review_reasons = set(frame_reasons) - {'profile', 'small_face'}
            results.append({
                'rects': mapped_rects,
                'landmarks': mapped_landmarks,
                'rect_scores': rect_scores,
                'rotation': entry['rotation'],
                'used_refinement': used_refinement,
                'used_full_refinement': used_full_refinement,
                'used_light_refinement': used_light_refinement,
                'quality_reasons': frame_reasons,
                'suspect': len(review_reasons) != 0 or used_full_refinement,
            })

        return results
