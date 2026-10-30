import operator
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms as torchvision_nms

from core.leras import nn

class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        # Initializing Leras/PyTorch environment
        nn.initialize(data_format="NCHW")
        
        # Model path
        model_path = Path(__file__).parent / "S3FD.npy"
        if not model_path.exists():
            raise Exception("Unable to load S3FD.npy")

        # Define L2Norm Layer
        class L2Norm(nn.LayerBase):
            def __init__(self, n_channels, **kwargs):
                self.n_channels = n_channels
                super().__init__(**kwargs)

            def build_weights(self):
                # TF: (1, 1, 1, C)
                # PyTorch NCHW: (1, C, 1, 1)
                self.weight = torch.nn.Parameter(torch.ones(1, self.n_channels, 1, 1, dtype=nn.floatx))

            def forward(self, x):
                # x: NCHW
                # norm over C (dim 1)
                norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-10
                return x / norm * self.weight

        # Define S3FD Model
        class S3FD(nn.ModelBase):
            def __init__(self):
                super().__init__(name='S3FD')

            def on_build(self):
                # self.minus = tf.constant([104,117,123], dtype=nn.floatx )
                # PyTorch: Register buffer for mean subtraction
                self.register_buffer("minus", torch.tensor([104, 117, 123], dtype=nn.floatx).view(1, 3, 1, 1))

                # Layers defined exactly as in TF version to match weights
                self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, strides=1, padding='SAME')
                self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding='SAME')

                self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, strides=1, padding='SAME')
                self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, strides=1, padding='SAME')

                self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')

                self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.fc6 = nn.Conv2D(512, 1024, kernel_size=3, strides=1, padding=3) # Padding 3 matches TF
                self.fc7 = nn.Conv2D(1024, 1024, kernel_size=1, strides=1, padding='SAME')

                self.conv6_1 = nn.Conv2D(1024, 256, kernel_size=1, strides=1, padding='SAME')
                self.conv6_2 = nn.Conv2D(256, 512, kernel_size=3, strides=2, padding='SAME')

                self.conv7_1 = nn.Conv2D(512, 128, kernel_size=1, strides=1, padding='SAME')
                self.conv7_2 = nn.Conv2D(128, 256, kernel_size=3, strides=2, padding='SAME')

                self.conv3_3_norm = L2Norm(256)
                self.conv4_3_norm = L2Norm(512)
                self.conv5_3_norm = L2Norm(512)

                self.conv3_3_norm_mbox_conf = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3_norm_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv4_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv5_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.fc7_mbox_conf = nn.Conv2D(1024, 2, kernel_size=3, strides=1, padding='SAME')
                self.fc7_mbox_loc = nn.Conv2D(1024, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv6_2_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv6_2_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv7_2_mbox_conf = nn.Conv2D(256, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv7_2_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

            def forward(self, inp):
                # inp is list of tensors, usually [x]
                if isinstance(inp, (list, tuple)):
                    x = inp[0]
                else:
                    x = inp

                # Preprocess: subtract minus. x is NCHW, minus is 1C11
                # If input is BGR, S3FD expects specific subtraction.
                # TF code: x = x - self.minus
                # self.minus was [104, 117, 123]
                x = x - self.minus

                x = F.relu(self.conv1_1(x))
                x = F.relu(self.conv1_2(x))
                x = F.max_pool2d(x, 2, 2) # VALID, pool 2x2 stride 2

                x = F.relu(self.conv2_1(x))
                x = F.relu(self.conv2_2(x))
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.conv3_1(x))
                x = F.relu(self.conv3_2(x))
                x = F.relu(self.conv3_3(x))
                f3_3 = x
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.conv4_1(x))
                x = F.relu(self.conv4_2(x))
                x = F.relu(self.conv4_3(x))
                f4_3 = x
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.conv5_1(x))
                x = F.relu(self.conv5_2(x))
                x = F.relu(self.conv5_3(x))
                f5_3 = x
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                ffc7 = x

                x = F.relu(self.conv6_1(x))
                x = F.relu(self.conv6_2(x))
                f6_2 = x

                x = F.relu(self.conv7_1(x))
                x = F.relu(self.conv7_2(x))
                f7_2 = x

                f3_3 = self.conv3_3_norm(f3_3)
                f4_3 = self.conv4_3_norm(f4_3)
                f5_3 = self.conv5_3_norm(f5_3)

                cls1 = self.conv3_3_norm_mbox_conf(f3_3)
                reg1 = self.conv3_3_norm_mbox_loc(f3_3)

                cls2 = self.conv4_3_norm_mbox_conf(f4_3)
                reg2 = self.conv4_3_norm_mbox_loc(f4_3)

                cls3 = self.conv5_3_norm_mbox_conf(f5_3)
                reg3 = self.conv5_3_norm_mbox_loc(f5_3)

                cls4 = self.fc7_mbox_conf(ffc7)
                reg4 = self.fc7_mbox_loc(ffc7)

                cls5 = self.conv6_2_mbox_conf(f6_2)
                reg5 = self.conv6_2_mbox_loc(f6_2)

                cls6 = self.conv7_2_mbox_conf(f7_2)
                reg6 = self.conv7_2_mbox_loc(f7_2)

                # The first scale maxes its three background channels before softmax.
                bmax = torch.max( torch.max(cls1[:, 0:1, :, :], cls1[:, 1:2, :, :]), cls1[:, 2:3, :, :] )
                cls1 = torch.cat( [bmax, cls1[:, 3:4, :, :]], dim=1 )
                
                # Softmax
                # TF: tf.nn.softmax(cls1) (default axis=-1)
                # Should be Softmax over channels?
                # TF Softmax on (N, H, W, C) implies axis=-1 (C).
                # PyTorch: dimension 1 is C.
                cls1 = F.softmax(cls1, dim=1)
                cls2 = F.softmax(cls2, dim=1)
                cls3 = F.softmax(cls3, dim=1)
                cls4 = F.softmax(cls4, dim=1)
                cls5 = F.softmax(cls5, dim=1)
                cls6 = F.softmax(cls6, dim=1)

                # Return list. 
                # Note: original returned NHWC numpy arrays.
                # We should return tensors here, and let ModelBase.run handle conversion?
                # But ModelBase run returns whatever run returns.
                # If we return NCHW, we must verify if 'refine' expects NHWC or NCHW.
                # 'refine' iterates ocls[hindex, windex, 1].
                # This implies ocls is (H, W, C).
                # So we MUST Permute outputs to NHWC before returning, 
                # OR ModelBase.run converts them?
                # ModelBase.run converts Tensors to Numpy. It maintains shapes.
                # So we must permute here to NHWC.
                
                results = [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]
                # Opt: Return NCHW directly to avoid permute cost
                return results

        # Initialize model
        self.model = S3FD()
        self.model.build() # Ensure layers are built and parameters registered
        if place_model_on_cpu:
            self.model = self.model.cpu()
        else:
            self.model = self.model.to(nn.device)

        # Load weights
        # weight load handled by ModelBase -> Saveable -> load_weights(path)
        # It handles TF mapping.
        if not self.model.load_weights(model_path):
             raise Exception("Failed to load S3FD weights.")
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
        
        # Optimization: Enable cuDNN benchmark for fixed size inputs (if useful)
        # S3FD inputs change size, but benchmark might help if sizes repeat.
        if not place_model_on_cpu:
             torch.backends.cudnn.benchmark = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False

    def extract (self, input_image, is_bgr=True, is_remove_intersects=False):
        # input_image: (H, W, C) numpy
        if is_bgr:
            input_image = input_image[:,:,::-1].copy()
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        new_w, new_h = int(w/input_scale), int(h/input_scale)

        if input_scale > 1.0:
            input_image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        elif not input_image.flags["C_CONTIGUOUS"]:
            input_image = np.ascontiguousarray(input_image)

        # Run inference (Optimized: Direct Tensor run, no_grad, FP16)
        
        # 1. Prepare Tensor on Device
        # Convert HWC numpy -> NCHW tensor
        input_tensor = torch.from_numpy(input_image).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        
        # Move to device
        device = self.device
        input_tensor = input_tensor.to(device)
        
        # 2. Forward Pass
        with torch.no_grad():
            use_amp = device.type == 'cuda'
            with torch.amp.autocast('cuda', enabled=use_amp):
                # self.model is S3FD(ModelBase). ModelBase is nn.Module.
                # Calling self.model(x) calls forward.
                # Our forward returns NCHW list of tensors (Optimized).
                olist = self.model( input_tensor )
        
        # 3. Refine on GPU
        detected_faces = self.refine(olist) # Returns (N, 5) numpy array (l,t,r,b,score)
        
        # 4. Scale back
        if len(detected_faces) > 0:
            # detected_faces is already numpy
            # Scale coordinates
            detected_faces[:, :4] *= input_scale
            
            # Post-Process (filtering checks from original code)
            final_faces = []
            for face_rect in detected_faces:
                 l,t,r,b,score = face_rect
                 bt = b-t
                 if min(r-l, bt) < 40:
                     continue
                 b += bt*0.1
                 final_faces.append( [int(l), int(t), int(r), int(b), score] )
            
            # Sort by area (largest first)
            final_faces.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            
            # Remove score for output (return list of [l,t,r,b])
            detected_faces = [ x[:4] for x in final_faces ]
        else:
            detected_faces = []

        if is_remove_intersects:
            # Original list-based intersection removal
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1 = detected_faces[i]
                l0,t0,r0,b0 = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)

        return detected_faces

    def _prepare_saenxt_input(self, input_image, target_size, is_bgr):
        h, w, _ = input_image.shape
        d = max(w, h)
        scale_to = max(64.0, min(float(target_size), float(d)))
        input_scale = d / scale_to
        new_w = max(1, int(w / input_scale))
        new_h = max(1, int(h / input_scale))

        if new_w != w or new_h != h:
            prepared = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            prepared = input_image

        # Resizing is channel-independent. Convert only the reduced detector
        # input instead of copying the complete source frame.
        if is_bgr:
            prepared = prepared[:, :, ::-1].copy()
        elif not prepared.flags["C_CONTIGUOUS"]:
            prepared = np.ascontiguousarray(prepared)

        return prepared, input_scale

    @staticmethod
    def _remove_intersecting_faces(detected_faces):
        for i in range(len(detected_faces) - 1, 0, -1):
            l1, t1, r1, b1 = detected_faces[i][:4]
            l0, t0, r0, b0 = detected_faces[i - 1][:4]
            dx = min(r0, r1) - max(l0, l1)
            dy = min(b0, b1) - max(t0, t1)
            if dx >= 0 and dy >= 0:
                detected_faces.pop(i)
        return detected_faces

    def _postprocess_saenxt_detections(self, detections, input_scale, is_remove_intersects):
        result = []
        if len(detections) != 0:
            detections = detections.copy()
            detections[:, :4] *= input_scale
            for l, t, r, b, score in detections:
                box_height = b - t
                if min(r - l, box_height) < 40:
                    continue
                b += box_height * 0.1
                result.append([int(l), int(t), int(r), int(b), float(score)])

        result.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        if is_remove_intersects:
            result = self._remove_intersecting_faces(result)
        return result

    def _run_saenxt_batch(self, prepared_images):
        batch = np.stack(prepared_images)
        input_tensor = torch.from_numpy(batch).to(self.device, non_blocking=self.device.type == 'cuda')
        if input_tensor.dtype == torch.uint8:
            input_tensor = input_tensor.float()
        elif input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.to(torch.float32)
        input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                return self.model(input_tensor)

    def extract_batch_saenxt(self,
                              input_images,
                              target_size=640,
                              is_bgr=True,
                              is_remove_intersects=False,
                              max_batch_size=None):
        """Batched detector path used only by the SAENXT extractor."""
        if len(input_images) == 0:
            return []

        prepared = []
        scales = []
        shape_groups = {}
        for image_idx, image in enumerate(input_images):
            detector_image, input_scale = self._prepare_saenxt_input(image, target_size, is_bgr)
            prepared.append(detector_image)
            scales.append(input_scale)
            shape_groups.setdefault(detector_image.shape[:2], []).append(image_idx)

        results = [[] for _ in input_images]
        requested_batch_size = max_batch_size or len(input_images)
        requested_batch_size = max(1, int(requested_batch_size))

        def process_indices(indices):
            if len(indices) == 0:
                return
            try:
                outputs = self._run_saenxt_batch([prepared[idx] for idx in indices])
                for batch_idx, image_idx in enumerate(indices):
                    detections = self.refine(outputs, batch_idx=batch_idx)
                    results[image_idx] = self._postprocess_saenxt_detections(
                        detections,
                        scales[image_idx],
                        is_remove_intersects,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in str(err).lower()
                if not is_oom or len(indices) == 1:
                    raise
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                midpoint = len(indices) // 2
                process_indices(indices[:midpoint])
                process_indices(indices[midpoint:])

        for indices in shape_groups.values():
            for start in range(0, len(indices), requested_batch_size):
                process_indices(indices[start:start + requested_batch_size])

        return results

    def extract_saenxt(self,
                       input_image,
                       target_size=640,
                       is_bgr=True,
                       is_remove_intersects=False):
        return self.extract_batch_saenxt(
            [input_image],
            target_size=target_size,
            is_bgr=is_bgr,
            is_remove_intersects=is_remove_intersects,
            max_batch_size=1,
        )[0]

    def refine(self, olist, batch_idx=0):
        # olist: List of Tensors [cls1, reg1, ...], shape (1, C, H, W) NCHW (Optimized)
        
        bboxlist = []
        
        # Iterate pairs
        for i, (ocls, oreg) in enumerate ( zip ( olist[::2], olist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            # ocls, oreg are Tensors (1, C, H, W)
            # Match DeepFaceLab-main detector pruning threshold.
            # ocls is (1, 2, H, W). Score is channel 1.
            
            score_map = ocls[batch_idx, 1, :, :] # (H, W)
            mask = score_map > 0.05
            
            if not torch.any(mask):
                continue
                
            # Get indices (h, w)
            # torch.where returns (h_idx, w_idx)
            hindex, windex = torch.where(mask)
            
            score = score_map[hindex, windex]
            
            # oreg is (1, 4, H, W).
            # loc needs to be (N, 4).
            # We extract columns at hindex, windex.
            # oreg[0, :, hindex, windex] -> Shape (4, N)
            loc = oreg[batch_idx, :, hindex, windex]
            loc = loc.t() # (N, 4)
            
            # Priors (N, 4)
            # Convert indices to float for calculation
            windex = windex.float()
            hindex = hindex.float()
            
            # Priors: [w*stride+s_d2, h*stride+s_d2, s_m4, s_m4]
            # We can use pure tensor ops
            
            priors = torch.stack([
                windex * stride + s_d2,
                hindex * stride + s_d2,
                torch.full_like(windex, s_m4),
                torch.full_like(windex, s_m4)
            ], dim=1)
            
            priors_2p = priors[:, 2:]
            
            # Decode center and size offsets from their priors.
            box_xy = priors[:, :2] + loc[:, :2] * 0.1 * priors_2p
            box_wh = priors_2p * torch.exp(loc[:, 2:] * 0.2)
            
            # Convert center-wh to xyxy (l, t, r, b)
            l_t = box_xy - box_wh / 2
            r_b = l_t + box_wh
            
            # Shape (N, 4)
            boxes = torch.cat([l_t, r_b], dim=1)
            
            # Append boxes and scores
            # keeping them separate or fused?
            # fused for concatenation
            
            layer_res = torch.cat([boxes, score.unsqueeze(1)], dim=1)
            bboxlist.append(layer_res)

        if len(bboxlist) == 0:
            return np.zeros((0, 5))
            
        # Concat all layers
        detections = torch.cat(bboxlist, dim=0)
        
        # NMS (TorchVision)
        # detections: (Total, 5) . [x1, y1, x2, y2, score]
        
        keep_indices = torchvision_nms(detections[:, :4], detections[:, 4], 0.3)
        detections = detections[keep_indices]
        
        # Filter score >= 0.5 (Final threshold from original code)
        mask_final = detections[:, 4] >= 0.5
        detections = detections[mask_final]
        
        # Return to CPU Numpy
        return detections.cpu().numpy()

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
