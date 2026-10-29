from dataclasses import asdict, dataclass

import numpy as np


MERGE_MODES = (
    "overlay",
    "natural",
    "lighting-match",
    "seamless",
    "hist-match",
    "raw-rgb",
    "raw-predict",
)

TWO_PASS_MODES = ("none", "face", "face+mask")

COLOR_TRANSFER_MODES = {
    "none": 0,
    "rct": 1,
    "lct": 2,
    "mkl": 3,
    "mkl-m": 4,
    "idt": 5,
    "idt-m": 6,
    "sot-m": 7,
    "mix-m": 8,
}

MASK_MODES = {
    "full": 0,
    "landmarks": 1,
    "learned-swap": 2,
    "learned-dst": 3,
    "learned": 4,
    "learned-union": 5,
    "xseg-swap": 6,
    "xseg-dst": 7,
    "xseg": 8,
    "learned-xseg": 9,
    "learned-minimum": 10,
}


def _bounded_int(value, low, high, default=0):
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        value = int(default)
    return int(np.clip(value, low, high))


def _effective_erode(control):
    control = _bounded_int(control, -100, 100)
    if control == 0:
        return 0
    amount = 25.0 * ((abs(control) / 100.0) ** 1.6)
    return int(round(-amount if control < 0 else amount))


def _effective_scale(control):
    control = _bounded_int(control, -100, 100)
    if control == 0:
        return 0
    amount = 18.0 * ((abs(control) / 100.0) ** 1.5)
    return int(round(-amount if control < 0 else amount))


@dataclass
class WebMergeSettings:
    settings_version: int = 1
    mode: str = "natural"
    mask_mode: str = "learned"
    two_pass: str = "none"
    color_transfer: str = "none"
    preview_view: str = "merged"

    two_pass_strength: int = 65
    erode: int = 0
    feather: int = 28
    transition: int = 18
    containment: int = 0
    matte_clean: int = 0
    edge_clean: int = 0
    edge_decontam: int = 0
    occlusion: int = 0
    output_scale: int = 0

    color_transfer_strength: int = 30
    hist_match_strength: int = 0
    pre_sharpen: int = 0
    pre_clahe: int = 0
    latent_sharpen: int = 0
    style_guidance: int = 100
    detail: int = 0
    sharpen: int = 0
    super_resolution: int = 0

    gamma: int = 0
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    red_offset: int = 0
    green_offset: int = 0
    blue_offset: int = 0

    restore_eyes: int = 0
    eyes_grow: int = 12
    eyes_feather: int = 10
    restore_mouth: int = 0
    mouth_grow: int = 14
    mouth_feather: int = 12

    motion_blur: int = 0
    temporal_stability: int = 0
    temporal_aa: int = 0

    @classmethod
    def from_dict(cls, payload):
        payload = payload if isinstance(payload, dict) else {}
        settings = cls()

        for key in settings.__dataclass_fields__:
            if key in payload:
                setattr(settings, key, payload[key])

        settings.settings_version = 1
        settings.mode = str(settings.mode).lower().strip()
        if settings.mode not in MERGE_MODES:
            settings.mode = "natural"

        settings.mask_mode = str(settings.mask_mode).lower().strip()
        if settings.mask_mode not in MASK_MODES:
            settings.mask_mode = "learned"

        settings.two_pass = str(settings.two_pass).lower().strip()
        if settings.two_pass not in TWO_PASS_MODES:
            settings.two_pass = "none"

        settings.color_transfer = str(settings.color_transfer).lower().strip()
        if settings.color_transfer not in COLOR_TRANSFER_MODES:
            settings.color_transfer = "none"

        settings.preview_view = str(settings.preview_view).lower().strip()
        if settings.preview_view not in ("merged", "original", "mask", "raw-swap"):
            settings.preview_view = "merged"

        for name in (
            "two_pass_strength",
            "containment",
            "matte_clean",
            "edge_clean",
            "edge_decontam",
            "occlusion",
            "color_transfer_strength",
            "hist_match_strength",
            "pre_clahe",
            "latent_sharpen",
            "detail",
            "super_resolution",
            "restore_eyes",
            "restore_mouth",
            "motion_blur",
            "temporal_stability",
            "temporal_aa",
        ):
            setattr(settings, name, _bounded_int(getattr(settings, name), 0, 100))

        settings.erode = _bounded_int(settings.erode, -100, 100)
        settings.feather = _bounded_int(settings.feather, 0, 400)
        settings.transition = _bounded_int(settings.transition, 0, 100)
        settings.output_scale = _bounded_int(settings.output_scale, -100, 100)
        settings.pre_sharpen = _bounded_int(settings.pre_sharpen, 0, 200)
        settings.style_guidance = _bounded_int(settings.style_guidance, 0, 200, 100)
        settings.sharpen = _bounded_int(settings.sharpen, -100, 100)

        for name in (
            "gamma",
            "brightness",
            "contrast",
            "saturation",
            "red_offset",
            "green_offset",
            "blue_offset",
        ):
            setattr(settings, name, _bounded_int(getattr(settings, name), -100, 100))

        settings.eyes_grow = _bounded_int(settings.eyes_grow, -40, 80)
        settings.eyes_feather = _bounded_int(settings.eyes_feather, 0, 80)
        settings.mouth_grow = _bounded_int(settings.mouth_grow, -40, 80)
        settings.mouth_feather = _bounded_int(settings.mouth_feather, 0, 80)
        if settings.mode == "hist-match" and settings.hist_match_strength <= 0:
            # hist-match with a 0% mix renders identically to overlay. Apply
            # the same auto-fill the page uses when the mode is picked so a
            # persisted 0 does not silently disable the mode.
            settings.hist_match_strength = 65
        return settings

    def to_dict(self):
        return asdict(self)

    def copy(self):
        return WebMergeSettings.from_dict(self.to_dict())

    def apply_to_config(self, config):
        config.mode = self.mode
        config.mask_mode = MASK_MODES[self.mask_mode]
        if getattr(config, "xseg_only", False) and config.mask_mode not in (6, 7, 8):
            config.mask_mode = 8

        config.two_pass_mode = TWO_PASS_MODES.index(self.two_pass)
        config.two_pass_strength = self.two_pass_strength
        config.erode_mask_modifier = _effective_erode(self.erode)
        config.blur_mask_modifier = self.feather
        config.output_face_scale = _effective_scale(self.output_scale)
        config.color_transfer_mode = COLOR_TRANSFER_MODES[self.color_transfer]
        config.super_resolution_power = self.super_resolution
        config.motion_blur_power = self.motion_blur
        config.pre_sharpen_mode = 1 if self.pre_sharpen > 0 else 0
        config.pre_sharpen_power = self.pre_sharpen
        config.sharpen_mode = 1
        config.blursharpen_amount = self.sharpen

        config.web_structured_matte = True
        config.transition_strength = self.transition
        config.containment_strength = self.containment
        config.matte_clean = self.matte_clean
        config.edge_clean = self.edge_clean
        config.edge_decontam = self.edge_decontam
        config.occlusion_strength = self.occlusion
        config.color_transfer_strength = self.color_transfer_strength
        config.hist_match_strength = self.hist_match_strength
        config.pre_clahe_amount = self.pre_clahe
        config.latent_sharpen = self.latent_sharpen
        config.style_guidance = self.style_guidance
        config.detail_amount = self.detail
        config.gamma = self.gamma
        config.brightness = self.brightness
        config.contrast = self.contrast
        config.saturation = self.saturation
        config.red_offset = self.red_offset
        config.green_offset = self.green_offset
        config.blue_offset = self.blue_offset
        config.restore_eyes_power = self.restore_eyes
        config.restore_eyes_size = self.eyes_grow
        config.restore_eyes_blur = self.eyes_feather
        config.restore_mouth_power = self.restore_mouth
        config.restore_mouth_size = self.mouth_grow
        config.restore_mouth_blur = self.mouth_feather
        config.temporal_stability = self.temporal_stability
        config.temporal_aa = self.temporal_aa
        config.preview_view = self.preview_view
        config.supports_guidance = True
        return config

