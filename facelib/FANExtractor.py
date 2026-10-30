import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla
import torch
import torch.nn.functional as F

from facelib import FaceType, LandmarksProcessor
from core.leras import nn

"""
ported from https://github.com/1adrianb/face-alignment
PyTorch port
"""
class FANExtractor(object):
    def __init__ (self, landmarks_3D=False, place_model_on_cpu=False):
        
        model_path = Path(__file__).parent / ( "2DFAN.npy" if not landmarks_3D else "3DFAN.npy")
        if not model_path.exists():
            raise Exception("Unable to load FANExtractor model")

        # Initialize Leras
        nn.initialize(data_format="NCHW")

        class ConvBlock(nn.ModelBase):
            def on_build(self, in_planes, out_planes):
                self.in_planes = in_planes
                self.out_planes = out_planes

                self.bn1 = nn.BatchNorm2D(in_planes)
                self.conv1 = nn.Conv2D (in_planes, out_planes//2, kernel_size=3, strides=1, padding='SAME', use_bias=False )

                self.bn2 = nn.BatchNorm2D(out_planes//2)
                self.conv2 = nn.Conv2D (out_planes//2, out_planes//4, kernel_size=3, strides=1, padding='SAME', use_bias=False )

                self.bn3 = nn.BatchNorm2D(out_planes//4)
                self.conv3 = nn.Conv2D (out_planes//4, out_planes//4, kernel_size=3, strides=1, padding='SAME', use_bias=False )

                if self.in_planes != self.out_planes:
                    self.down_bn1 = nn.BatchNorm2D(in_planes)
                    self.down_conv1 = nn.Conv2D (in_planes, out_planes, kernel_size=1, strides=1, padding='VALID', use_bias=False )
                else:
                    self.down_bn1 = None
                    self.down_conv1 = None

            def forward(self, input):
                x = input
                x = self.bn1(x)
                x = F.relu(x)
                x = out1 = self.conv1(x)

                x = self.bn2(x)
                x = F.relu(x)
                x = out2 = self.conv2(x)

                x = self.bn3(x)
                x = F.relu(x)
                x = out3 = self.conv3(x)

                x = torch.cat([out1, out2, out3], dim=1) # Dim 1 is Channels in NCHW

                if self.in_planes != self.out_planes:
                    downsample = self.down_bn1(input)
                    downsample = F.relu (downsample)
                    downsample = self.down_conv1 (downsample)
                    x = x + downsample
                else:
                    x = x + input

                return x

        class HourGlass (nn.ModelBase):
            def on_build(self, in_planes, depth):
                self.b1 = ConvBlock (in_planes, 256)
                self.b2 = ConvBlock (in_planes, 256)

                if depth > 1:
                    self.b2_plus = HourGlass(256, depth-1)
                else:
                    self.b2_plus = ConvBlock(256, 256)

                self.b3 = ConvBlock(256, 256)

            def forward(self, input):
                up1 = self.b1(input)

                low1 = F.avg_pool2d(input, 2, 2) # stride 2, pool 2
                low1 = self.b2 (low1)

                low2 = self.b2_plus(low1)
                low3 = self.b3(low2)

                # up2 = nn.upsample2d(low3) -> Nearest neighbor upscale
                up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

                return up1+up2

        class FAN (nn.ModelBase):
            def __init__(self):
                super().__init__(name='FAN')

            def on_build(self):
                self.conv1 = nn.Conv2D (3, 64, kernel_size=7, strides=2, padding='SAME') # 7x7 stride 2 SAME
                self.bn1 = nn.BatchNorm2D(64)

                self.conv2 = ConvBlock(64, 128)
                self.conv3 = ConvBlock(128, 128)
                self.conv4 = ConvBlock(128, 256)

                self.m = []
                self.top_m = []
                self.conv_last = []
                self.bn_end = []
                self.l = []
                self.bl = []
                self.al = []
                for i in range(4):
                    self.m += [ HourGlass(256, 4) ]
                    self.top_m += [ ConvBlock(256, 256) ]

                    self.conv_last += [ nn.Conv2D (256, 256, kernel_size=1, strides=1, padding='VALID') ]
                    self.bn_end += [ nn.BatchNorm2D(256) ]

                    self.l += [ nn.Conv2D (256, 68, kernel_size=1, strides=1, padding='VALID') ]

                    if i < 4-1:
                        self.bl += [ nn.Conv2D (256, 256, kernel_size=1, strides=1, padding='VALID') ]
                        self.al += [ nn.Conv2D (68, 256, kernel_size=1, strides=1, padding='VALID') ]

            def forward(self, inp) :
                # inp can be list
                if isinstance(inp, (list, tuple)):
                    x = inp[0]
                else:
                    x = inp

                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)

                x = self.conv2(x)
                x = F.avg_pool2d(x, 2, 2)
                x = self.conv3(x)
                x = self.conv4(x)

                outputs = []
                previous = x
                for i in range(4):
                    ll = self.m[i] (previous)
                    ll = self.top_m[i] (ll)
                    ll = self.conv_last[i] (ll)
                    ll = self.bn_end[i] (ll)
                    ll = F.relu(ll)
                    tmp_out = self.l[i](ll)
                    outputs.append(tmp_out)
                    if i < 4 - 1:
                        ll = self.bl[i](ll)
                        previous = previous + ll + self.al[i](tmp_out)
                x = outputs[-1]
                # x is NCHW.
                # TF code transposed (0,3,1,2) -> NCHW from NHWC?
                # or NHWC from NCHW?
                # TF `Conv2D` outputs NHWC.
                # `get_pts` expects `a_ch, a_h, a_w`. (C, H, W).
                # So if TF tranposed to (0,3,1,2), it made it NCHW (0=N, 3=C, 1=H, 2=W).
                # PyTorch x is already NCHW.
                return x

        # Initialize model
        self.model = FAN()
        self.model.build()
        if place_model_on_cpu:
            self.model = self.model.cpu()
        else:
            self.model = self.model.to(nn.device)
             
        self.model.load_weights(str(model_path))
        self.model.eval()
        self.device = next(self.model.parameters()).device

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def _run_batch(self, images):
        if not images.flags["C_CONTIGUOUS"]:
            images = np.ascontiguousarray(images)

        x = torch.from_numpy(images)
        if x.dtype == torch.uint8:
            x = x.to(self.device, non_blocking=True).float().div_(255.0)
        else:
            x = x.to(self.device, non_blocking=True)
            if x.dtype != torch.float32:
                x = x.float()

        x = x.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            use_amp = self.device.type == 'cuda'
            with torch.amp.autocast('cuda', enabled=use_amp):
                predicted = self.model(x)

        return predicted.float().cpu().numpy()

    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:,:,::-1].copy()
            is_bgr = False

        rect_predictions = {i: [] for i in range(len(rects))}
        all_crops = []
        crop_metadata = []

        for rect_idx, (left, top, right, bottom) in enumerate(rects):
            scale = (right - left + bottom - top) / 195.0

            center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
            centers = [ center ]

            if multi_sample:
                centers += [ center + [-1,-1],
                             center + [1,-1],
                             center + [1,1],
                             center + [-1,1],
                           ]

            try:
                for c in centers:
                    all_crops.append(self.crop(input_image, c, scale))
                    crop_metadata.append((rect_idx, c.copy(), scale))
            except:
                traceback.print_exc()

        if len(all_crops) == 0:
            landmarks = [None] * len(rects)
        else:
            predicted = self._run_batch(np.stack(all_crops))

            for pred, (rect_idx, center, scale) in zip(predicted, crop_metadata):
                rect_predictions[rect_idx].append(self.get_pts_from_predict(pred, center, scale))

            landmarks = []
            for rect_idx in range(len(rects)):
                ptss = rect_predictions[rect_idx]
                if len(ptss) != 0:
                    landmarks.append(np.mean(np.array(ptss), 0))
                else:
                    landmarks.append(None)

        if second_pass_extractor is not None:
            for i, lmrks in enumerate(landmarks):
                try:
                    if lmrks is not None:
                        image_to_face_mat = LandmarksProcessor.get_transform_mat (lmrks, 256, FaceType.FULL)
                        face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC )

                        rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                        if len(rects2) == 1: #dont do second pass if faces != 1 detected in cropped image
                            lmrks2 = self.extract (face_image, [ rects2[0] ], is_bgr=is_bgr, multi_sample=True)[0]
                            landmarks[i] = LandmarksProcessor.transform_points (lmrks2, image_to_face_mat, True)
                except:
                    pass

        return landmarks

    @staticmethod
    def _saenxt_heatmap_metrics(prediction):
        channels, height, width = prediction.shape
        flat = prediction.reshape(channels, height * width)
        peaks = np.max(flat, axis=1)
        means = np.mean(flat, axis=1)
        stds = np.std(flat, axis=1)
        peak_z = (peaks - means) / np.maximum(stds, 1e-6)

        peak_indices = np.argmax(flat, axis=1)
        peak_x = peak_indices % width
        peak_y = peak_indices // width
        edge_hits = (peak_x <= 0) | (peak_x >= width - 1) | (peak_y <= 0) | (peak_y >= height - 1)

        if flat.shape[1] > 1:
            second_peaks = np.partition(flat, -2, axis=1)[:, -2]
            peak_margin = peaks - second_peaks
        else:
            peak_margin = peaks

        return {
            'peak_z_median': float(np.median(peak_z)),
            'peak_z_min': float(np.min(peak_z)),
            'peak_margin_median': float(np.median(peak_margin)),
            'edge_fraction': float(np.mean(edge_hits)),
        }

    def _run_batch_chunked(self, images, max_batch_size):
        max_batch_size = max(1, int(max_batch_size or len(images)))
        outputs = []

        def run_chunk(chunk):
            try:
                outputs.append(self._run_batch(np.stack(chunk)))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in str(err).lower()
                if not is_oom or len(chunk) == 1:
                    raise
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                midpoint = len(chunk) // 2
                run_chunk(chunk[:midpoint])
                run_chunk(chunk[midpoint:])

        for start in range(0, len(images), max_batch_size):
            run_chunk(images[start:start + max_batch_size])

        return np.concatenate(outputs, axis=0)

    def extract_batch_saenxt(self, items, multi_sample=False, max_batch_size=None):
        """Batch landmarks across frames without copying complete source images."""
        landmarks_result = [[None] * len(rects) for _, rects in items]
        metrics_result = [[None] * len(rects) for _, rects in items]
        predictions_by_rect = {}
        metrics_by_rect = {}
        all_crops = []
        crop_metadata = []

        for item_idx, (input_image, rects) in enumerate(items):
            for rect_idx, (left, top, right, bottom) in enumerate(rects):
                scale = (right - left + bottom - top) / 195.0
                center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
                centers = [center]
                if multi_sample:
                    centers += [center + [-1, -1],
                                center + [1, -1],
                                center + [1, 1],
                                center + [-1, 1]]

                try:
                    for sample_center in centers:
                        crop = self.crop(input_image, sample_center, scale)
                        crop = crop[:, :, ::-1].copy()
                        all_crops.append(crop)
                        crop_metadata.append((item_idx, rect_idx, sample_center.copy(), scale))
                except Exception:
                    traceback.print_exc()

        if len(all_crops) == 0:
            return landmarks_result, metrics_result

        predicted = self._run_batch_chunked(all_crops, max_batch_size)
        for pred, (item_idx, rect_idx, center, scale) in zip(predicted, crop_metadata):
            key = (item_idx, rect_idx)
            predictions_by_rect.setdefault(key, []).append(self.get_pts_from_predict(pred, center, scale))
            metrics_by_rect.setdefault(key, []).append(self._saenxt_heatmap_metrics(pred))

        for (item_idx, rect_idx), point_samples in predictions_by_rect.items():
            points = np.asarray(point_samples, dtype=np.float32)
            mean_points = np.mean(points, axis=0)
            landmarks_result[item_idx][rect_idx] = mean_points

            sample_metrics = metrics_by_rect[(item_idx, rect_idx)]
            rect = items[item_idx][1][rect_idx]
            rect_size = max(1.0, float(max(rect[2] - rect[0], rect[3] - rect[1])))
            if len(points) > 1:
                spread = np.linalg.norm(points - mean_points[None, ...], axis=2)
                sample_spread = float(np.median(spread) / rect_size)
            else:
                sample_spread = 0.0

            metrics_result[item_idx][rect_idx] = {
                'peak_z_median': float(np.mean([m['peak_z_median'] for m in sample_metrics])),
                'peak_z_min': float(np.min([m['peak_z_min'] for m in sample_metrics])),
                'peak_margin_median': float(np.mean([m['peak_margin_median'] for m in sample_metrics])),
                'edge_fraction': float(np.max([m['edge_fraction'] for m in sample_metrics])),
                'sample_spread': sample_spread,
            }

        return landmarks_result, metrics_result

    def transform(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype( int )
        br = self.transform([resolution, resolution], center, scale, resolution).astype( int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        # a: (C, H, W)
        a_ch, a_h, a_w = a.shape

        b = a.reshape ( (a_ch, a_h*a_w) )
        c = b.argmax(1).reshape ( (a_ch, 1) ).repeat(2, axis=1).astype(np.float64) # float compatibility
        c[:,0] %= a_w
        c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a_w), 0, c[:,1] )

        for i in range(a_ch):
            pX, pY = int(c[i,0]), int(c[i,1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
                c[i] += np.sign(diff)*0.25

        c += 0.5

        return np.array( [ self.transform (c[i], center, scale, a_w) for i in range(a_ch) ] )
