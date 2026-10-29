"""Automatic low-overhead training incident reports."""

from __future__ import annotations

import datetime
import faulthandler
import os
import platform
import subprocess
import sys
import threading
import traceback
from pathlib import Path


_SENSITIVE_OPTION_PARTS = ("password", "passwd", "secret", "token", "api_key", "apikey")


def _safe_call(callback, default="unavailable"):
    try:
        return callback()
    except Exception:
        return default


def _safe_repr(value, limit=4000):
    try:
        text = repr(value)
    except Exception as error:
        text = f"<repr failed: {type(error).__name__}: {error}>"
    if len(text) > limit:
        return text[:limit] + f"... <truncated {len(text) - limit} characters>"
    return text


def _append_section(lines, title, values):
    lines.extend(("", f"===== {title} ====="))
    if isinstance(values, str):
        lines.append(values)
    else:
        lines.extend(str(value) for value in values)


def _model_lines(model):
    if model is None:
        return ["Model was not constructed when the incident occurred."]

    lines = [
        f"class: {model.__class__.__module__}.{model.__class__.__name__}",
        f"name: {_safe_call(lambda: model.get_model_name(), getattr(model, 'model_name', 'unknown'))}",
        f"iteration: {_safe_call(lambda: model.get_iter(), getattr(model, 'iter', 'unknown'))}",
        f"batch_size: {_safe_call(lambda: model.get_batch_size(), getattr(model, 'batch_size', 'unknown'))}",
    ]
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        lines.extend(
            (
                f"trainer: {trainer.__class__.__module__}.{trainer.__class__.__name__}",
                f"device: {_safe_repr(getattr(trainer, 'device', 'unknown'))}",
                f"fp16: {_safe_repr(getattr(trainer, 'use_fp16', 'unknown'))}",
                f"amp_dtype: {_safe_repr(getattr(trainer, 'amp_dtype', 'unknown'))}",
                f"architecture: {_safe_repr(getattr(trainer, 'archi', 'unknown'))}",
            )
        )
        scaler = getattr(trainer, "scaler", None)
        if scaler is not None:
            lines.append(f"grad_scaler_scale: {_safe_call(lambda: scaler.get_scale())}")
        optimizer = getattr(trainer, "optim_G", None)
        if optimizer is not None:
            lines.append(f"generator_optimizer: {optimizer.__class__.__name__}")
            for index, group in enumerate(getattr(optimizer, "param_groups", [])):
                lines.append(
                    f"optimizer_group_{index}: lr={_safe_repr(group.get('lr'))}, "
                    f"clipnorm={_safe_repr(group.get('clipnorm'))}, "
                    f"weight_decay={_safe_repr(group.get('weight_decay'))}"
                )
    return lines


def _option_lines(model):
    options = getattr(model, "options", None) if model is not None else None
    if not isinstance(options, dict):
        return ["No model options were available."]

    lines = []
    for key in sorted(options, key=lambda item: str(item).casefold()):
        key_text = str(key)
        value_text = "<redacted>" if any(
            part in key_text.casefold() for part in _SENSITIVE_OPTION_PARTS
        ) else _safe_repr(options[key])
        lines.append(f"{key_text}: {value_text}")
    return lines


def _recent_loss_lines(model, count=20):
    if model is None:
        return ["No loss history was available."]
    history = _safe_call(lambda: model.get_loss_history(), [])
    if not history:
        return ["No completed iterations were present in loss history."]
    start = max(0, len(history) - count)
    return [f"iteration {index}: {_safe_repr(history[index])}" for index in range(start, len(history))]


def _sample_lines(model):
    filenames_by_side = getattr(model, "last_sample_filenames", None) if model is not None else None
    if not filenames_by_side:
        return ["No last-sample filenames were available."]

    lines = []
    labels = ("SRC", "DST", "TEMPORAL")
    for index, filenames in enumerate(filenames_by_side):
        label = labels[index] if index < len(labels) else f"GROUP_{index}"
        lines.append(f"{label}:")
        lines.extend(f"  {filename}" for filename in filenames) if filenames else lines.append("  <empty>")
    return lines


def _torch_lines():
    try:
        import torch
    except Exception as error:
        return [f"PyTorch diagnostics unavailable: {type(error).__name__}: {error}"]

    lines = [
        f"torch_version: {torch.__version__}",
        f"torch_cuda_build: {getattr(torch.version, 'cuda', None)}",
        f"cudnn_version: {_safe_call(lambda: torch.backends.cudnn.version())}",
        f"cuda_available: {_safe_call(lambda: torch.cuda.is_available(), False)}",
    ]
    if not _safe_call(lambda: torch.cuda.is_available(), False):
        return lines

    device_count = _safe_call(lambda: torch.cuda.device_count(), 0)
    lines.append(f"cuda_device_count: {device_count}")
    for index in range(int(device_count or 0)):
        lines.extend(
            (
                f"device_{index}_name: {_safe_call(lambda i=index: torch.cuda.get_device_name(i))}",
                f"device_{index}_capability: {_safe_call(lambda i=index: torch.cuda.get_device_capability(i))}",
            )
        )

    current = _safe_call(lambda: torch.cuda.current_device(), None)
    lines.append(f"current_cuda_device: {current}")
    if current is not None:
        free_total = _safe_call(lambda: torch.cuda.mem_get_info(current), None)
        if free_total is not None:
            free_bytes, total_bytes = free_total
            lines.append(f"cuda_memory_free_mb: {free_bytes / (1024 ** 2):.1f}")
            lines.append(f"cuda_memory_total_mb: {total_bytes / (1024 ** 2):.1f}")
        lines.extend(
            (
                f"cuda_allocated_mb: {_safe_call(lambda: torch.cuda.memory_allocated(current) / (1024 ** 2))}",
                f"cuda_reserved_mb: {_safe_call(lambda: torch.cuda.memory_reserved(current) / (1024 ** 2))}",
                f"cuda_peak_allocated_mb: {_safe_call(lambda: torch.cuda.max_memory_allocated(current) / (1024 ** 2))}",
                f"cuda_peak_reserved_mb: {_safe_call(lambda: torch.cuda.max_memory_reserved(current) / (1024 ** 2))}",
            )
        )
    memory_summary = _safe_call(
        lambda: torch.cuda.memory_summary(device=current, abbreviated=False), None
    )
    if memory_summary:
        lines.extend(("", "--- torch.cuda.memory_summary ---", memory_summary))
    return lines


def _nvidia_smi_lines():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,temperature.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as error:
        return [f"nvidia-smi unavailable: {type(error).__name__}: {error}"]
    lines = [f"exit_code: {result.returncode}"]
    if result.stdout.strip():
        lines.extend(result.stdout.strip().splitlines())
    if result.stderr.strip():
        lines.extend(("stderr:", *result.stderr.strip().splitlines()))
    return lines


def _thread_lines():
    frames = sys._current_frames()
    thread_names = {thread.ident: thread.name for thread in threading.enumerate()}
    lines = []
    for thread_id, frame in frames.items():
        lines.append(f"--- thread {thread_names.get(thread_id, 'unknown')} ({thread_id}) ---")
        lines.extend(line.rstrip("\n") for line in traceback.format_stack(frame, limit=60))
    return lines or ["No Python thread stacks were available."]


def write_training_report(
    saved_models_path,
    incident_type,
    *,
    model=None,
    exc_info=None,
    message=None,
    details=None,
):
    """Write a complete incident report and return its path, never raising."""
    try:
        report_dir = Path(saved_models_path) / "crash_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now().astimezone()
        iteration = _safe_call(lambda: int(model.get_iter()), None) if model is not None else None
        iteration_text = f"_iter{iteration:06d}" if isinstance(iteration, int) else ""
        safe_type = "".join(
            character if character.isalnum() else "_" for character in str(incident_type)
        ).strip("_") or "incident"
        report_path = report_dir / (
            f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{safe_type}{iteration_text}.txt"
        )

        lines = [
            "SAENXT AUTOMATIC TRAINING INCIDENT REPORT",
            f"incident_type: {incident_type}",
            f"local_time: {now.isoformat()}",
            f"utc_time: {now.astimezone(datetime.timezone.utc).isoformat()}",
            f"process_id: {os.getpid()}",
            f"python: {sys.version.replace(os.linesep, ' ')}",
            f"platform: {platform.platform()}",
            f"working_directory: {os.getcwd()}",
            f"command_line: {_safe_repr(sys.argv)}",
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}",
            f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<not set>')}",
        ]
        if message:
            _append_section(lines, "MESSAGE", str(message))
        if details:
            _append_section(
                lines,
                "INCIDENT DETAILS",
                [f"{key}: {_safe_repr(value)}" for key, value in sorted(details.items())],
            )

        if exc_info and exc_info[0] is not None:
            exception_lines = [
                f"class: {exc_info[0].__module__}.{exc_info[0].__name__}",
                f"message: {exc_info[1]}",
                "",
                *[line.rstrip("\n") for line in traceback.format_exception(*exc_info)],
            ]
        else:
            exception_lines = [
                "No Python exception was raised. The reporting call stack follows:",
                *[line.rstrip("\n") for line in traceback.format_stack(limit=60)],
            ]
        _append_section(lines, "EXCEPTION OR DETECTION STACK", exception_lines)
        _append_section(lines, "MODEL", _model_lines(model))
        _append_section(lines, "LAST SAMPLE FILENAMES", _sample_lines(model))
        _append_section(lines, "RECENT COMPLETED LOSSES", _recent_loss_lines(model))
        _append_section(lines, "ACTIVE MODEL OPTIONS", _option_lines(model))
        _append_section(lines, "PYTORCH AND CUDA", _torch_lines())
        _append_section(lines, "NVIDIA-SMI", _nvidia_smi_lines())
        _append_section(lines, "ALL PYTHON THREAD STACKS", _thread_lines())
        lines.extend(("", "===== END REPORT =====", ""))

        payload = "\n".join(lines)
        temp_path = report_path.with_suffix(report_path.suffix + ".tmp")
        temp_path.write_text(payload, encoding="utf-8", errors="replace")
        os.replace(temp_path, report_path)
        latest_path = report_dir / "LATEST_TRAINING_INCIDENT.txt"
        latest_temp = latest_path.with_suffix(latest_path.suffix + ".tmp")
        latest_temp.write_text(payload, encoding="utf-8", errors="replace")
        os.replace(latest_temp, latest_path)
        return report_path
    except Exception as error:
        try:
            print(f"[CrashReport] Failed to write report: {type(error).__name__}: {error}")
        except Exception:
            pass
        return None


class FatalCrashLog:
    """Faulthandler target that remains only when the process exits abnormally."""

    def __init__(self, saved_models_path):
        self.handle = None
        self.path = None
        self.enabled = False
        try:
            report_dir = Path(saved_models_path) / "crash_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.datetime.now().astimezone()
            self.path = report_dir / (
                f"ACTIVE_FATAL_{now.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.txt"
            )
            self.handle = open(self.path, "w", encoding="utf-8", buffering=1)
            self.handle.write("SAENXT FATAL PROCESS CRASH LOG\n")
            self.handle.write(f"started: {now.isoformat()}\n")
            self.handle.write(f"process_id: {os.getpid()}\n")
            self.handle.write(
                "If this file remains after training exits, the process did not shut down cleanly.\n\n"
            )
            self.handle.flush()
            faulthandler.enable(file=self.handle, all_threads=True)
            self.enabled = True
        except Exception:
            self.close(clean=True)

    def close(self, clean=True):
        if self.enabled:
            try:
                faulthandler.disable()
            except Exception:
                pass
            self.enabled = False
        if self.handle is not None:
            try:
                self.handle.flush()
                self.handle.close()
            except Exception:
                pass
            self.handle = None
        if clean and self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass


def start_fatal_crash_log(saved_models_path):
    return FatalCrashLog(saved_models_path)
