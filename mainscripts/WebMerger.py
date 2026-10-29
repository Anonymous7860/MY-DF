import json
import logging
import math
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import numpy.linalg as npla

import samplelib
from core import imagelib, pathex
from core.cv2ex import cv2_imread, cv2_imwrite
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from facelib.xseg_options import resolve_xseg_runtime_options
from merger import FrameInfo, MergeMasked, MergerConfig, TemporalMergeState
from merger.WebMergeSettings import WebMergeSettings


class _QuietPollingLogFilter(logging.Filter):
    def filter(self, record):
        return "/api/merge/progress " not in record.getMessage()


def _choose_destination_folder(default_path):
    script_path = Path(__file__).parents[2] / "select_merge_folder.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Destination folder picker not found: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        "--default-path",
        str(default_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "The destination folder picker could not be opened."
        raise RuntimeError(detail)

    selected_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return Path(selected_lines[-1]) if selected_lines else None


def _log_xseg_options_error(model_data_path, err):
    io.log_info(
        f"Warning: unable to read {model_data_path.name}; merger XSeg will use current 256. "
        f"Error: {err}"
    )


def _iter_aligned_images(aligned_path):
    packed_samples = None
    try:
        packed_samples = samplelib.PackedFaceset.load(aligned_path)
    except Exception:
        io.log_err(
            f"Error occurred while loading PackedFaceset from {aligned_path}: "
            f"{traceback.format_exc()}"
        )

    if packed_samples is not None:
        io.log_info("Using packed faceset.")
        for sample in io.progress_bar_generator(packed_samples, "Collecting alignments"):
            filepath = Path(sample.filename)
            yield filepath, DFLIMG.load(
                filepath,
                loader_func=lambda _, packed_sample=sample: packed_sample.read_raw_file(),
            )
        return

    for filepath in io.progress_bar_generator(
        pathex.get_image_paths(aligned_path),
        "Collecting alignments",
    ):
        filepath = Path(filepath)
        yield filepath, DFLIMG.load(filepath)


def _collect_frames(input_path, aligned_path):
    if aligned_path is None or not aligned_path.exists():
        raise FileNotFoundError("Aligned directory not found.")

    alignments = {}
    for filepath, dflimg in _iter_aligned_images(aligned_path):
        if dflimg is None or not dflimg.has_data():
            io.log_err(f"{filepath.name} is not a DFL image file")
            continue
        source_filename = dflimg.get_source_filename()
        if source_filename is None:
            continue
        source_stem = Path(source_filename).stem
        alignments.setdefault(source_stem, []).append(
            (dflimg.get_source_landmarks(), dflimg)
        )

    frames = []
    for filename in pathex.get_image_paths(input_path):
        filepath = Path(filename)
        aligned = alignments.get(filepath.stem, ())
        frames.append(
            FrameInfo(
                filepath=filepath,
                landmarks_list=[entry[0] for entry in aligned],
                dfl_images_list=[entry[1] for entry in aligned],
            )
        )

    _compute_motion_vectors(frames)
    return frames


def _compute_motion_vectors(frames):
    if not frames or any(len(frame.landmarks_list) > 1 for frame in frames):
        return
    size = 256
    local_points = [(size // 2 - 1, size // 2 - 1), (size // 2 - 1, 0)]
    for index, frame in enumerate(frames):
        previous = frames[max(0, index - 1)]
        following = frames[min(len(frames) - 1, index + 1)]
        if not previous.landmarks_list or not frame.landmarks_list or not following.landmarks_list:
            continue
        previous_mat = LandmarksProcessor.get_transform_mat(
            previous.landmarks_list[0],
            size,
            face_type=FaceType.FULL,
        )
        next_mat = LandmarksProcessor.get_transform_mat(
            following.landmarks_list[0],
            size,
            face_type=FaceType.FULL,
        )
        previous_points = LandmarksProcessor.transform_points(local_points, previous_mat, True)
        next_points = LandmarksProcessor.transform_points(local_points, next_mat, True)
        motion_vector = next_points[0] - previous_points[0]
        frame.motion_power = float(npla.norm(motion_vector))
        if frame.motion_power > 0:
            motion_vector = motion_vector / frame.motion_power
            frame.motion_deg = float(
                -math.atan2(motion_vector[1], motion_vector[0]) * 180.0 / math.pi
            )


class WebMergerServer:
    def __init__(
        self,
        *,
        model,
        predictor_func,
        predictor_input_shape,
        base_config,
        frames,
        face_enhancer_func,
        xseg_extract_func,
        xseg_input_size,
        xseg_weights_available,
        input_path=None,
        output_path,
        output_mask_path,
        settings_path,
        folder_selector=None,
        frame_collector=None,
        port=8081,
    ):
        from flask import Flask

        template_dir = Path(__file__).parent.parent / "merger" / "templates"
        gfx_dir = Path(__file__).parent.parent / "gfx"
        self.app = Flask(
            "saenxt_web_merger",
            template_folder=str(template_dir),
            static_folder=str(gfx_dir),
            static_url_path="/gfx",
        )
        self.model = model
        self.predictor_func = predictor_func
        self.predictor_input_shape = predictor_input_shape
        self.base_config = base_config
        self.frames = frames
        self.face_enhancer_func = face_enhancer_func
        self.xseg_extract_func = xseg_extract_func
        self.xseg_input_size = int(xseg_input_size)
        self.xseg_weights_available = bool(xseg_weights_available)
        self.input_path = Path(input_path) if input_path is not None else None
        self.output_path = Path(output_path)
        self.output_mask_path = Path(output_mask_path)
        self.settings_path = Path(settings_path)
        self._folder_selector = folder_selector or _choose_destination_folder
        self._frame_collector = frame_collector or _collect_frames
        self.port = int(port)

        self.architecture = str(model.options.get("archi", "")).lower().strip()
        from models.Model_SAENXT.core.config import SUPPORTED_ARCHIS

        self.supported_architectures = tuple(SUPPORTED_ARCHIS)
        if self.architecture not in self.supported_architectures:
            raise ValueError(
                f"Unsupported SAENXT architecture '{self.architecture}'. "
                f"Supported: {', '.join(self.supported_architectures)}"
            )

        self.settings = self._load_settings()
        self._settings_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._destination_lock = threading.RLock()
        self._preview_cache = OrderedDict()
        self._preview_cache_bytes = 0
        self._preview_cache_budget = 256 * 1024 * 1024
        # Flask serves requests from several threads: the cache needs its own
        # lock, and the destination generation keeps a render that was already
        # in flight when the folder changed from being cached or served
        # against the new folder's frames.
        self._cache_lock = threading.Lock()
        self._destination_generation = 0
        self._batch_lock = threading.Lock()
        # Look-ahead renderer for playback / stepping. Interactive requests
        # always win: the worker yields between frames while one is waiting.
        self._direct_lock = threading.Lock()
        self._direct_waiting = 0
        self._prefetch_lock = threading.Lock()
        self._prefetch_request = None
        self._prefetch_wake = threading.Event()
        self._prefetch_thread = None
        self._prefetch_active_key = None
        self._batch_thread = None
        self._batch_stop = threading.Event()
        self._progress_lock = threading.RLock()
        self._progress = {
            "running": False,
            "stopping": False,
            "completed": 0,
            "total": len(self.frames),
            "percent": 0.0,
            "fps": 0.0,
            "eta_seconds": None,
            "current_frame": "",
            "error": None,
            "finished": False,
        }
        self._register_routes()

    def _load_settings(self):
        if not self.settings_path.exists():
            return WebMergeSettings()
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return WebMergeSettings.from_dict(payload)
        except Exception as err:
            io.log_info(
                f"Warning: unable to read {self.settings_path.name}; using Web Merger defaults. "
                f"Error: {err}"
            )
            return WebMergeSettings()

    def _save_settings(self):
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(
            json.dumps(self.settings.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _settings_signature(self, settings):
        return json.dumps(
            settings.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def _clear_preview_cache(self):
        with self._cache_lock:
            self._preview_cache.clear()
            self._preview_cache_bytes = 0

    def _invalidate_rendered_previews(self):
        """Drop merged/mask/raw-swap previews; source frames stay valid."""
        with self._cache_lock:
            for key in [k for k in self._preview_cache if k[2] != "original"]:
                self._preview_cache_bytes -= len(self._preview_cache.pop(key))

    def _stable_frame_info(self, index, strength):
        source = self.frames[index]
        strength = float(np.clip(strength, 0, 100)) / 100.0
        if strength <= 0.0 or not source.landmarks_list:
            return source

        previous = self.frames[max(0, index - 1)]
        following = self.frames[min(len(self.frames) - 1, index + 1)]
        if not (
            len(previous.landmarks_list)
            == len(source.landmarks_list)
            == len(following.landmarks_list)
        ):
            return source

        landmarks = []
        for prev_points, current_points, next_points in zip(
            previous.landmarks_list,
            source.landmarks_list,
            following.landmarks_list,
        ):
            current = np.asarray(current_points, dtype=np.float32)
            smoothed = (
                np.asarray(prev_points, dtype=np.float32)
                + current * 2.0
                + np.asarray(next_points, dtype=np.float32)
            ) / 4.0
            landmarks.append(current * (1.0 - strength) + smoothed * strength)

        frame = FrameInfo(
            filepath=source.filepath,
            landmarks_list=landmarks,
            dfl_images_list=source.dfl_images_list,
        )
        frame.motion_power = source.motion_power
        frame.motion_deg = source.motion_deg
        return frame

    def _render_rgba(self, index, settings):
        if index < 0 or index >= len(self.frames):
            raise IndexError("Frame index is outside the available range.")

        frame = self._stable_frame_info(index, settings.temporal_stability)
        if not frame.landmarks_list:
            # MergeMasked reads the frame itself, so only the landmark-free
            # fallback needs its own disk read here.
            original = cv2_imread(frame.filepath)
            original = imagelib.normalize_channels(original, 3)
            mask = np.zeros(original.shape[:2] + (1,), dtype=np.uint8)
            return np.concatenate((original, mask), axis=-1)

        config = self.base_config.copy()
        settings.apply_to_config(config)
        with self._inference_lock:
            return MergeMasked(
                self.predictor_func,
                self.predictor_input_shape,
                face_enhancer_func=self.face_enhancer_func,
                xseg_extract_func=self.xseg_extract_func,
                xseg_input_size=self.xseg_input_size,
                xseg_weights_available=self.xseg_weights_available,
                cfg=config,
                frame_info=frame,
            )

    PREVIEW_JPEG_QUALITY = 90

    @classmethod
    def _encode_preview(cls, image):
        # JPEG encodes several times faster than PNG on full frames and is a
        # fraction of the size, which is what playback streaming is bound by.
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(cls.PREVIEW_JPEG_QUALITY)],
        )
        if not success:
            raise RuntimeError("OpenCV could not encode the preview.")
        return encoded.tobytes()

    def _preview_key(self, generation, index, view, settings):
        if view == "original":
            # The source frame does not depend on the merge settings, so a
            # settings tweak must not throw it out of the cache.
            return (generation, index, view, "")
        return (generation, index, view, self._settings_signature(settings))

    def _cache_get(self, key):
        with self._cache_lock:
            cached = self._preview_cache.get(key)
            if cached is not None:
                self._preview_cache.move_to_end(key)
            return cached

    def _cache_put(self, key, generation, payload):
        with self._cache_lock:
            if generation != self._destination_generation:
                return
            previous = self._preview_cache.pop(key, None)
            if previous is not None:
                self._preview_cache_bytes -= len(previous)
            self._preview_cache[key] = payload
            self._preview_cache_bytes += len(payload)
            while (
                len(self._preview_cache) > 1
                and self._preview_cache_bytes > self._preview_cache_budget
            ):
                _, evicted = self._preview_cache.popitem(last=False)
                self._preview_cache_bytes -= len(evicted)

    def _render_preview(self, index, view, settings):
        if view == "original":
            image = cv2_imread(self.frames[index].filepath)
            image = imagelib.normalize_channels(image, 3)
        else:
            render_settings = settings.copy()
            if view == "raw-swap":
                render_settings.mode = "raw-predict"
            rgba = self._render_rgba(index, render_settings)
            if view == "mask":
                alpha = rgba[..., 3]
                image = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
            else:
                image = rgba[..., :3]
        return self._encode_preview(image)

    def _preview_bytes(self, index, view, ahead=0):
        """Encoded preview for one frame; optionally warms `ahead` frames after it."""
        with self._settings_lock:
            settings = self.settings.copy()
        view = str(view or settings.preview_view).lower().strip()
        generation = self._destination_generation
        key = self._preview_key(generation, index, view, settings)
        cached = self._cache_get(key)
        if cached is None and self._prefetch_active_key == key:
            # The look-ahead worker is already rendering exactly this frame.
            deadline = time.monotonic() + 60.0
            while self._prefetch_active_key == key and time.monotonic() < deadline:
                time.sleep(0.01)
            cached = self._cache_get(key)
        if cached is None:
            with self._direct_lock:
                self._direct_waiting += 1
            try:
                cached = self._render_preview(index, view, settings)
            finally:
                with self._direct_lock:
                    self._direct_waiting -= 1
            self._cache_put(key, generation, cached)
        if ahead > 0 and view != "original" and not self._batch_running():
            self._schedule_prefetch(generation, index + 1, ahead, view, settings)
        return cached

    def _batch_running(self):
        return self._batch_thread is not None and self._batch_thread.is_alive()

    def _schedule_prefetch(self, generation, start, count, view, settings):
        if start >= len(self.frames):
            return
        with self._prefetch_lock:
            self._prefetch_request = (generation, start, count, view, settings)
            self._prefetch_wake.set()
            if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker,
                    name="SAENXT-WebMerger-Prefetch",
                    daemon=True,
                )
                self._prefetch_thread.start()

    def _prefetch_worker(self):
        while True:
            self._prefetch_wake.wait()
            with self._prefetch_lock:
                request = self._prefetch_request
                self._prefetch_request = None
                self._prefetch_wake.clear()
            if request is None:
                continue
            generation, start, count, view, settings = request
            signature = self._settings_signature(settings)
            for index in range(start, start + count):
                if self._prefetch_request is not None:
                    break  # a newer window supersedes this one
                if generation != self._destination_generation or self._batch_running():
                    break
                with self._settings_lock:
                    if self._settings_signature(self.settings) != signature:
                        break
                if index >= len(self.frames):
                    break
                key = self._preview_key(generation, index, view, settings)
                if self._cache_get(key) is not None:
                    continue
                # Let an interactive render go first.
                while self._direct_waiting > 0 and self._prefetch_request is None:
                    time.sleep(0.005)
                if self._prefetch_request is not None:
                    break
                self._prefetch_active_key = key
                try:
                    payload = self._render_preview(index, view, settings)
                except Exception:
                    break
                finally:
                    self._prefetch_active_key = None
                self._cache_put(key, generation, payload)

    def _state_payload(self):
        with self._settings_lock:
            settings = self.settings.to_dict()
        return {
            "architecture": self.architecture,
            "supported_architectures": list(self.supported_architectures),
            "model_name": self.model.get_model_name(),
            "iteration": int(self.model.get_iter()),
            "resolution": int(self.predictor_input_shape[0]),
            "frames": len(self.frames),
            "frame_names": [frame.filepath.name for frame in self.frames],
            "first_frame": self.frames[0].filepath.name if self.frames else None,
            "last_frame": self.frames[-1].filepath.name if self.frames else None,
            "input_path": str(self.input_path) if self.input_path is not None else None,
            "xseg_available": self.xseg_weights_available,
            "xseg_resolution": self.xseg_input_size,
            "output_path": str(self.output_path),
            "output_mask_path": str(self.output_mask_path),
            "settings": settings,
        }

    def _register_routes(self):
        from flask import jsonify, make_response, render_template, request, send_file
        import io as stdlib_io

        @self.app.after_request
        def disable_cache(response):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @self.app.route("/")
        def index():
            return make_response(render_template("web_merger.html"))

        @self.app.route("/api/state")
        def state():
            return jsonify(self._state_payload())

        @self.app.route("/api/destination/select", methods=["POST"])
        def select_destination():
            with self._batch_lock:
                if self._batch_thread is not None and self._batch_thread.is_alive():
                    return jsonify({"error": "Stop the active batch before changing destination folders."}), 409
            try:
                selected = self._folder_selector(self.input_path)
                if selected is None:
                    return jsonify({"ok": True, "cancelled": True})

                selected = Path(selected)
                aligned_path = selected / "aligned"
                frames = self._frame_collector(selected, aligned_path)
                if not frames:
                    raise ValueError("No image frames were found in the selected destination folder.")
                if not any(frame.landmarks_list for frame in frames):
                    raise ValueError(
                        "The selected frames do not match any source filenames in the aligned faceset."
                    )

                with self._destination_lock:
                    self.input_path = selected
                    self.output_path = selected / "merged"
                    self.output_mask_path = selected / "merged_mask"
                    self.frames = frames
                    self._destination_generation += 1
                    self._clear_preview_cache()
                    with self._progress_lock:
                        self._progress.update(
                            running=False,
                            stopping=False,
                            completed=0,
                            total=len(frames),
                            percent=0.0,
                            fps=0.0,
                            eta_seconds=None,
                            current_frame="",
                            error=None,
                            finished=False,
                        )
                return jsonify({"ok": True, "cancelled": False, "state": self._state_payload()})
            except Exception as err:
                return jsonify({"error": str(err)}), 400

        @self.app.route("/api/frame/<int:index>/preview")
        def preview(index):
            try:
                ahead = max(0, min(32, int(request.args.get("ahead", 0) or 0)))
                payload = self._preview_bytes(
                    index,
                    request.args.get("view", "merged"),
                    ahead=ahead,
                )
                return send_file(
                    stdlib_io.BytesIO(payload),
                    mimetype="image/jpeg",
                    max_age=0,
                )
            except Exception as err:
                # Surface the real failure in the console; the browser's image
                # loader can only show a generic "could not be rendered" toast.
                print(f"[WEB MERGER] Preview render failed (frame {index}):")
                traceback.print_exc()
                return jsonify({"error": str(err)}), 400

        @self.app.route("/api/settings", methods=["POST"])
        def update_settings():
            payload = request.get_json(silent=True) or {}
            with self._settings_lock:
                self.settings = WebMergeSettings.from_dict(payload)
                self._save_settings()
                settings = self.settings.to_dict()
            self._invalidate_rendered_previews()
            return jsonify({"ok": True, "settings": settings})

        @self.app.route("/api/settings/reset", methods=["POST"])
        def reset_settings():
            with self._settings_lock:
                self.settings = WebMergeSettings()
                self._save_settings()
                settings = self.settings.to_dict()
            self._invalidate_rendered_previews()
            return jsonify({"ok": True, "settings": settings})

        @self.app.route("/api/merge/start", methods=["POST"])
        def start_merge():
            if not self.frames:
                return jsonify({"error": "Choose a destination folder before starting the merge."}), 400
            with self._batch_lock:
                if self._batch_thread is not None and self._batch_thread.is_alive():
                    return jsonify({"error": "A batch merge is already running."}), 409
                self._batch_stop.clear()
                self._batch_thread = threading.Thread(
                    target=self._run_batch,
                    name="SAENXT-WebMerger-Batch",
                    daemon=True,
                )
                self._batch_thread.start()
            return jsonify({"ok": True})

        @self.app.route("/api/merge/stop", methods=["POST"])
        def stop_merge():
            self._batch_stop.set()
            with self._progress_lock:
                self._progress["stopping"] = True
            return jsonify({"ok": True})

        @self.app.route("/api/merge/progress")
        def progress():
            with self._progress_lock:
                return jsonify(dict(self._progress))

    def _run_batch(self):
        started = time.perf_counter()
        with self._settings_lock:
            settings = self.settings.copy()
        temporal_state = (
            TemporalMergeState(strength=min(0.95, settings.temporal_aa / 100.0 * 0.95))
            if settings.temporal_aa > 0
            else None
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_mask_path.mkdir(parents=True, exist_ok=True)
        with self._progress_lock:
            self._progress.update(
                running=True,
                stopping=False,
                completed=0,
                total=len(self.frames),
                percent=0.0,
                fps=0.0,
                eta_seconds=None,
                current_frame="",
                error=None,
                finished=False,
            )

        # PNG encoding plus the two disk writes per frame are pure CPU/disk
        # work; running them on a small writer pool lets the next frame's GPU
        # inference proceed while the previous frame is still being written.
        def _write_frame(name, image, mask):
            cv2_imwrite(self.output_path / name, image)
            cv2_imwrite(self.output_mask_path / name, mask[..., None])

        pending_writes = deque()

        def _drain_writes(limit):
            while len(pending_writes) > limit:
                pending_writes.popleft().result()

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="merge-writer") as writer_pool:
                for index, frame in enumerate(self.frames):
                    if self._batch_stop.is_set():
                        break
                    rgba = self._render_rgba(index, settings)
                    image = rgba[..., :3]
                    mask = rgba[..., 3]

                    if temporal_state is not None:
                        original = cv2_imread(frame.filepath)
                        original = imagelib.normalize_channels(original, 3).astype(np.float32) / 255.0
                        image_float = image.astype(np.float32) / 255.0
                        mask_float = mask.astype(np.float32) / 255.0
                        image_float, mask_float = temporal_state.process(
                            index,
                            original,
                            image_float,
                            mask_float,
                        )
                        image = np.rint(np.clip(image_float, 0.0, 1.0) * 255.0).astype(np.uint8)
                        mask = np.rint(np.clip(mask_float, 0.0, 1.0) * 255.0).astype(np.uint8)

                    pending_writes.append(
                        writer_pool.submit(_write_frame, frame.filepath.name, image, mask)
                    )
                    # A bounded queue keeps RAM flat; .result() re-raises any
                    # write error on this thread so it lands in progress state.
                    _drain_writes(4)

                    completed = index + 1
                    elapsed = max(1e-6, time.perf_counter() - started)
                    fps = completed / elapsed
                    remaining = len(self.frames) - completed
                    with self._progress_lock:
                        self._progress.update(
                            completed=completed,
                            percent=100.0 * completed / max(1, len(self.frames)),
                            fps=fps,
                            eta_seconds=remaining / max(1e-6, fps),
                            current_frame=frame.filepath.name,
                        )
                _drain_writes(0)
        except Exception:
            with self._progress_lock:
                self._progress["error"] = traceback.format_exc()
        finally:
            with self._progress_lock:
                self._progress["running"] = False
                self._progress["stopping"] = False
                self._progress["finished"] = (
                    self._progress["error"] is None
                    and not self._batch_stop.is_set()
                    and self._progress["completed"] == len(self.frames)
                )

    def run(self):
        url = f"http://127.0.0.1:{self.port}"
        io.log_info(f"SAENXT Web Merger: {url}")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        request_logger = logging.getLogger("werkzeug")
        polling_filter = _QuietPollingLogFilter()
        request_logger.addFilter(polling_filter)
        # Keep the console to our own line: Werkzeug's "development server"
        # warning and per-request access log are info-level on this logger,
        # and Flask prints its "Serving Flask app" banner through click.
        previous_level = request_logger.level
        request_logger.setLevel(logging.ERROR)
        import flask.cli as flask_cli
        previous_banner = flask_cli.show_server_banner
        flask_cli.show_server_banner = lambda *args, **kwargs: None
        try:
            self.app.run(
                host="127.0.0.1",
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False,
            )
        finally:
            request_logger.removeFilter(polling_filter)
            request_logger.setLevel(previous_level)
            flask_cli.show_server_banner = previous_banner


def main(
    model_class_name=None,
    saved_models_path=None,
    force_model_name=None,
    input_path=None,
    output_path=None,
    output_mask_path=None,
    aligned_path=None,
    force_gpu_idxs=None,
    cpu_only=None,
    port=8081,
):
    model = None
    try:
        if saved_models_path is None or not Path(saved_models_path).exists():
            raise FileNotFoundError(f"Model directory not found: {saved_models_path}")

        input_path = Path(input_path) if input_path is not None else Path.cwd()
        output_path = Path(output_path)
        output_mask_path = Path(output_mask_path)

        import models

        model = models.import_model(model_class_name)(
            is_training=False,
            saved_models_path=Path(saved_models_path),
            force_gpu_idxs=force_gpu_idxs,
            force_model_name=force_model_name,
            cpu_only=cpu_only,
        )
        predictor_func, predictor_input_shape, config = model.get_MergerConfig()
        if config.type != MergerConfig.TYPE_MASKED:
            raise ValueError("SAENXT Web Merger supports masked face models only.")

        run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
        xseg_archi, xseg_input_size = resolve_xseg_runtime_options(
            saved_models_path,
            on_error=_log_xseg_options_error,
        )
        xseg_weights_available = (
            Path(saved_models_path) / f"XSeg_{xseg_input_size}.npy"
        ).is_file()
        xseg_extract_func = MPClassFuncOnDemand(
            XSegNet,
            "extract",
            name="XSeg",
            resolution=xseg_input_size,
            archi=xseg_archi,
            weights_file_root=saved_models_path,
            place_model_on_cpu=True,
            run_on_cpu=run_on_cpu,
        )
        face_enhancer_func = MPClassFuncOnDemand(
            FaceEnhancer,
            "enhance",
            place_model_on_cpu=True,
            run_on_cpu=run_on_cpu,
        )

        settings_path = Path(
            model.get_strpath_storage_for_file("web_merger_settings.json")
        )
        server = WebMergerServer(
            model=model,
            predictor_func=predictor_func,
            predictor_input_shape=predictor_input_shape,
            base_config=config,
            frames=[],
            face_enhancer_func=face_enhancer_func,
            xseg_extract_func=xseg_extract_func,
            xseg_input_size=xseg_input_size,
            xseg_weights_available=xseg_weights_available,
            input_path=input_path,
            output_path=output_path,
            output_mask_path=output_mask_path,
            settings_path=settings_path,
            port=port,
        )
        server.run()
    except Exception:
        io.log_err(traceback.format_exc())
    finally:
        if model is not None:
            model.finalize()
