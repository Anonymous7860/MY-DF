# -*- coding: utf-8 -*-
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
"""
SAEHD (TensorFlow pickle) -> SAENXT (PyTorch) Weight Converter

Converts pretrained SAEHD LIAE-UDT weights from TF pickle format (.npy)
to PyTorch state_dict format (.pth) compatible with SAENXT.

What gets transferred:
  - encoder   (SAEHD encoder = 1:1 architecture match with SAEHDEncoder)
  - inter_B   (Dense bottleneck -- identical architecture)
  - decoder   (Upscale+Residual -- identical architecture)

What gets SKIPPED:
  - inter_AB  (Not present in pretrain weights, or reset on pretrain->train transition)
  - optimizer (Must restart fresh)
  - GAN       (Must restart fresh)

TF->PyTorch weight conversions:
  - Conv2D:  TF (H,W,I,O) -> PyTorch (O,I,H,W)
  - Dense:   TF (I,O) -> PyTorch (O,I)
  - Bias:    TF (N,) -> PyTorch (N,)  [no change]
  - No PixelShuffle channel reordering needed -- decoder now uses
    tf_depth_to_space which matches TF's native channel ordering.
   
Decoder key mapping:
  TF: "upscale0/conv1/weight:0" -> PyTorch: "upscale0.conv1.weight"
  TF: "res0/conv1/weight:0"    -> PyTorch: "res0.conv1.weight"
"""

import pickle
import numpy as np
import sys
import os
import re
from pathlib import Path

# Add DFL to path for torch import
dfl_root = Path(__file__).parent
sys.path.insert(0, str(dfl_root))

import torch


def load_tf_pickle(filepath):
    """Load TF-era DFL pickle weight file.

    Classic DFL Saveable files can be either a single dict or a stream where
    the first pickle is a weight count followed by one dict per variable.
    """
    with open(filepath, 'rb') as f:
        first_item = pickle.load(f)
        if isinstance(first_item, int):
            data = {}
            for _ in range(first_item):
                data.update(pickle.load(f))
        else:
            data = first_item
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    return data


def detect_weight_format(weight):
    """Detect the layout of a four-dimensional checkpoint tensor."""
    if len(weight.shape) != 4:
        return "unknown"

    d0, d1, d2, d3 = weight.shape

    if d0 <= 7 and d1 <= 7 and d3 >= d2:
        if abs(d0 - d1) <= 2:
            return "nhwc"

    if d2 <= 7 and d3 <= 7 and d0 >= d2:
        if abs(d2 - d3) <= 2:
            return "nchw"

    if d0 <= 7 and d1 <= 7:
        return "nhwc"
    if d2 <= 7 and d3 <= 7:
        return "nchw"

    return "unknown"


def detect_checkpoint_format(weights):
    """Detect the dominant tensor layout in a checkpoint."""
    nhwc_count = 0
    nchw_count = 0

    for weight in weights.values():
        if isinstance(weight, np.ndarray) and len(weight.shape) == 4:
            fmt = detect_weight_format(weight)
            if fmt == "nhwc":
                nhwc_count += 1
            elif fmt == "nchw":
                nchw_count += 1

    if nhwc_count > nchw_count:
        return "nhwc"
    if nchw_count > nhwc_count:
        return "nchw"
    return "unknown"


def normalize_layer_name(name):
    """Normalize a TensorFlow-style checkpoint layer name."""
    if ":" in name:
        name = name.split(":")[0]

    if name.startswith("ext/"):
        name = "ext_" + name[4:]

    return name.replace("/", ".")


def convert_weight_to_pytorch(weight, source_format="auto", layer_name=""):
    """Convert one checkpoint tensor to its PyTorch layout."""
    if not isinstance(weight, np.ndarray):
        return weight

    ndim = len(weight.shape)

    if ndim == 4:
        if source_format == "auto":
            source_format = detect_weight_format(weight)

        if source_format == "nhwc":
            return np.transpose(weight, (3, 2, 0, 1))
        if source_format == "nchw":
            return weight
        return np.transpose(weight, (3, 2, 0, 1))

    if ndim == 2:
        return np.transpose(weight, (1, 0))

    if ndim == 1:
        is_frn = ".frn.weight" in layer_name or ".frn.bias" in layer_name
        is_frn = is_frn or layer_name.endswith("frn/weight") or layer_name.endswith("frn/bias")
        if is_frn:
            return weight.reshape(1, -1, 1, 1)

    return weight


def tensor_from_legacy_weight(weight, pt_key, source_format):
    if isinstance(weight, np.ndarray):
        weight = convert_weight_to_pytorch(weight, source_format, pt_key)
        return torch.from_numpy(weight.copy())
    return weight


# The decoder uses TensorFlow-compatible depth-to-space channel ordering, so
# conversion does not reorder pixel-shuffle channels.


def convert_encoder(tf_weights):
    """
    Convert SAEHD '-t' encoder weights from TF pickle to PyTorch SAEHDEncoder state_dict.
    
    TF key mapping (1:1 match):
      down1/conv1/weight:0  -> down1.conv1.weight     Conv2D (H,W,I,O) -> (O,I,H,W)
      down1/conv1/bias:0    -> down1.conv1.bias
      res1/conv1/weight:0   -> res1.conv1.weight
      res1/conv1/bias:0     -> res1.conv1.bias
      res1/conv2/weight:0   -> res1.conv2.weight
      res1/conv2/bias:0     -> res1.conv2.bias
      down2/conv1/weight:0  -> down2.conv1.weight
      down2/conv1/bias:0    -> down2.conv1.bias
      down3/conv1/weight:0  -> down3.conv1.weight
      down3/conv1/bias:0    -> down3.conv1.bias
      down4/conv1/weight:0  -> down4.conv1.weight
      down4/conv1/bias:0    -> down4.conv1.bias
      down5/conv1/weight:0  -> down5.conv1.weight
      down5/conv1/bias:0    -> down5.conv1.bias
      res5/conv1/weight:0   -> res5.conv1.weight
      res5/conv1/bias:0     -> res5.conv1.bias
      res5/conv2/weight:0   -> res5.conv2.weight
      res5/conv2/bias:0     -> res5.conv2.bias
    """
    state_dict = {}
    
    mapping = {
        # Downscale layers: exact classic Downscale wrapper with conv1 sublayer.
        'down1/conv1/weight:0': ('down1.conv1.weight', 'conv'),
        'down1/conv1/bias:0':   ('down1.conv1.bias',   'bias'),
        'down2/conv1/weight:0': ('down2.conv1.weight', 'conv'),
        'down2/conv1/bias:0':   ('down2.conv1.bias',   'bias'),
        'down3/conv1/weight:0': ('down3.conv1.weight', 'conv'),
        'down3/conv1/bias:0':   ('down3.conv1.bias',   'bias'),
        'down4/conv1/weight:0': ('down4.conv1.weight', 'conv'),
        'down4/conv1/bias:0':   ('down4.conv1.bias',   'bias'),
        'down5/conv1/weight:0': ('down5.conv1.weight', 'conv'),
        'down5/conv1/bias:0':   ('down5.conv1.bias',   'bias'),
        # ResidualBlock layers
        'res1/conv1/weight:0':  ('res1.conv1.weight', 'conv'),
        'res1/conv1/bias:0':    ('res1.conv1.bias',   'bias'),
        'res1/conv2/weight:0':  ('res1.conv2.weight', 'conv'),
        'res1/conv2/bias:0':    ('res1.conv2.bias',   'bias'),
        'res5/conv1/weight:0':  ('res5.conv1.weight', 'conv'),
        'res5/conv1/bias:0':    ('res5.conv1.bias',   'bias'),
        'res5/conv2/weight:0':  ('res5.conv2.weight', 'conv'),
        'res5/conv2/bias:0':    ('res5.conv2.bias',   'bias'),
    }
    
    source_format = detect_checkpoint_format(tf_weights)
    normalized_weights = {
        normalize_layer_name(tf_key): (tf_key, weight)
        for tf_key, weight in tf_weights.items()
    }
    
    for tf_key, (pt_key, _kind) in mapping.items():
        normalized_key = normalize_layer_name(tf_key)
        if normalized_key not in normalized_weights:
            print(f"  WARNING: Missing key {tf_key}")
            continue
        
        src_key, w = normalized_weights[normalized_key]
        state_dict[pt_key] = tensor_from_legacy_weight(w, pt_key, source_format)
        print(f"  {src_key} {tf_weights[src_key].shape} -> {pt_key} {state_dict[pt_key].shape}")
    
    return state_dict


def convert_inter(tf_weights):
    """
    Convert Inter (bottleneck) weights from TF pickle to PyTorch state_dict.
    
    TF keys:
      dense1/weight:0  shape=(25088, 512)   -> dense1.weight  shape=(512, 25088)
      dense1/bias:0    shape=(512,)         -> dense1.bias    shape=(512,)
      dense2/weight:0  shape=(512, 50176)   -> dense2.weight  shape=(50176, 512)
      dense2/bias:0    shape=(50176,)       -> dense2.bias    shape=(50176,)
    """
    state_dict = {}
    
    mapping = {
        'dense1/weight:0': ('dense1.weight', 'dense'),
        'dense1/bias:0':   ('dense1.bias',   'bias'),
        'dense2/weight:0': ('dense2.weight', 'dense'),
        'dense2/bias:0':   ('dense2.bias',   'bias'),
    }
    
    source_format = detect_checkpoint_format(tf_weights)
    normalized_weights = {
        normalize_layer_name(tf_key): (tf_key, weight)
        for tf_key, weight in tf_weights.items()
    }
    
    for tf_key, (pt_key, _kind) in mapping.items():
        normalized_key = normalize_layer_name(tf_key)
        if normalized_key not in normalized_weights:
            print(f"  WARNING: Missing key {tf_key}")
            continue
        
        src_key, w = normalized_weights[normalized_key]
        state_dict[pt_key] = tensor_from_legacy_weight(w, pt_key, source_format)
        print(f"  {src_key} {tf_weights[src_key].shape} -> {pt_key} {state_dict[pt_key].shape}")
    
    return state_dict


def convert_decoder(tf_weights):
    """
    Convert Decoder weights from TF pickle to PyTorch state_dict.
    
    Key mapping (TF → PyTorch):
      upscale0/conv1/weight:0 → upscale0.conv1.weight  (Conv2D conversion)
      upscale0/conv1/bias:0   → upscale0.conv1.bias
      res0/conv1/weight:0     → res0.conv1.weight      (Conv2D conversion)
      res0/conv1/bias:0       → res0.conv1.bias
      res0/conv2/weight:0     → res0.conv2.weight      (Conv2D conversion)
      res0/conv2/bias:0       → res0.conv2.bias
      out_conv/weight:0       → out_conv.weight         (Conv2D conversion)
      out_conv/bias:0         → out_conv.bias
      upscalem0/conv1/weight:0 → upscalem0.conv1.weight
      out_convm/weight:0      → out_convm.weight
    """
    state_dict = {}
    source_format = detect_checkpoint_format(tf_weights)

    # Normalize each source name and convert tensors according to the detected
    # checkpoint layout. The classic base path retains its decoder layer names.
    for tf_key, w in tf_weights.items():
        pt_key = normalize_layer_name(tf_key)
        if pt_key.count(".") not in (1, 2):
            print(f"  SKIP: Unknown key format: {tf_key}")
            continue

        state_dict[pt_key] = tensor_from_legacy_weight(w, pt_key, source_format)
        print(f"  {tf_key} {tf_weights[tf_key].shape} -> {pt_key} {state_dict[pt_key].shape}")

    return state_dict

        # Convert weight format — just NHWC->NCHW transpose for convs, I,O->O,I for dense
        # No pixel shuffle channel reordering needed (decoder uses tf_depth_to_space)
def detect_tf_saehd_archi(model_dir, src_prefix, saehd_opts=None):
    """
    Detect whether the source TensorFlow SAEHD weights are LIAE or DF.

    Detection order:
      1. use archi from SAEHD options if present
      2. infer from expected TF weight filenames
      3. default to liae (legacy converter behavior)
    """
    if saehd_opts is not None:
        raw_archi = str(saehd_opts.get('archi', '')).lower().strip()
        if raw_archi.startswith('df'):
            return 'df'
        if raw_archi.startswith('liae'):
            return 'liae'

    has_df = all(
        (model_dir / f"{src_prefix}_{name}.npy").exists()
        for name in ('encoder', 'inter', 'decoder_src', 'decoder_dst')
    )
    has_liae = (
        (model_dir / f"{src_prefix}_encoder.npy").exists() and
        (model_dir / f"{src_prefix}_inter_B.npy").exists() and
        (model_dir / f"{src_prefix}_decoder.npy").exists()
    )

    if has_df and not has_liae:
        return 'df'
    if has_liae and not has_df:
        return 'liae'
    if has_df:
        return 'df'
    return 'liae'


def discover_saehd_prefix(model_dir):
    """
    Find the most complete source model prefix ending in '_SAEHD'.

    Examples:
      Pretrained_SAEHD_encoder.npy -> Pretrained_SAEHD
      Foo_SAEHD_data.dat          -> Foo_SAEHD
    """
    pattern = re.compile(
        r"^(?P<prefix>.+_SAEHD)_(?P<component>"
        r"encoder|inter|inter_B|inter_AB|decoder|decoder_src|decoder_dst|src_dst_opt"
        r")\.npy$|^(?P<data_prefix>.+_SAEHD)_(?P<data_component>data|summary)\.(?:dat|txt)$"
    )

    candidates = {}
    for path in model_dir.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue

        prefix = match.group('prefix') or match.group('data_prefix')
        component = match.group('component') or match.group('data_component')
        candidates.setdefault(prefix, set()).add(component)

    if not candidates:
        raise FileNotFoundError(
            f"No *_SAEHD weight set found in {model_dir}. "
            "Expected files like <name>_SAEHD_encoder.npy."
        )

    core_components = (
        'encoder', 'inter', 'inter_B', 'inter_AB',
        'decoder', 'decoder_src', 'decoder_dst', 'data'
    )

    best_prefix, _ = max(
        candidates.items(),
        key=lambda item: (
            sum(component in item[1] for component in core_components),
            len(item[1]),
            item[0],
        ),
    )

    if len(candidates) > 1:
        print("Found multiple *_SAEHD prefixes:")
        for prefix, components in sorted(candidates.items()):
            print(f"  - {prefix} ({', '.join(sorted(components))})")
        print(f"Using best match: {best_prefix}")

    return best_prefix


def main():
    print("=" * 70)
    print("  SAEHD (TF pickle) -> SAENXT (PyTorch) Weight Converter")
    print("=" * 70)
    
    # Paths
    model_dir = Path(__file__).parent.parent.parent / "workspace" / "model"

    if len(sys.argv) > 1:
        src_prefix = sys.argv[1]
        if not src_prefix.endswith("_SAEHD"):
            raise ValueError(
                f"Source prefix must end with '_SAEHD', got: {src_prefix}"
            )
    else:
        src_prefix = discover_saehd_prefix(model_dir)

    dst_prefix = src_prefix.removesuffix("SAEHD") + "SAENXT"
    
    print(f"\nSource:  {model_dir / src_prefix}_*.npy")
    print(f"Output:  {model_dir / dst_prefix}_*.pth")

    data_src = model_dir / f"{src_prefix}_data.dat"
    saehd_opts = {}
    if data_src.exists():
        try:
            saehd_opts = pickle.loads(data_src.read_bytes()).get('options', {})
        except Exception:
            saehd_opts = {}

    src_archi = detect_tf_saehd_archi(model_dir, src_prefix, saehd_opts)
    dst_archi = f"{src_archi}-udt"
    print(f"Detected source architecture: {src_archi}")
    print()
    
    # ===== CONVERT ENCODER =====
    encoder_path = model_dir / f"{src_prefix}_encoder.npy"
    if encoder_path.exists():
        print("[1/4] Converting encoder (SAEHD -> SAEHDEncoder 1:1 port)...")
        tf_encoder = load_tf_pickle(encoder_path)
        pt_encoder = convert_encoder(tf_encoder)
        
        out_path = model_dir / f"{dst_prefix}_encoder.pth"
        torch.save(pt_encoder, out_path)
        print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"[1/4] SKIP: {encoder_path.name} not found")
    
    if src_archi == 'df':
        inter_path = model_dir / f"{src_prefix}_inter.npy"
        if inter_path.exists():
            print("\n[2/4] Converting inter...")
            tf_inter = load_tf_pickle(inter_path)
            pt_inter = convert_inter(tf_inter)

            out_path = model_dir / f"{dst_prefix}_inter.pth"
            torch.save(pt_inter, out_path)
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"[2/4] SKIP: {inter_path.name} not found")

        decoder_src_path = model_dir / f"{src_prefix}_decoder_src.npy"
        if decoder_src_path.exists():
            print("\n[3/4] Converting decoder_src...")
            tf_decoder_src = load_tf_pickle(decoder_src_path)
            pt_decoder_src = convert_decoder(tf_decoder_src)

            out_path = model_dir / f"{dst_prefix}_decoder_src.pth"
            torch.save(pt_decoder_src, out_path)
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"\n[3/4] SKIP: {decoder_src_path.name} not found")

        decoder_dst_path = model_dir / f"{src_prefix}_decoder_dst.npy"
        if decoder_dst_path.exists():
            print("\n[4/4] Converting decoder_dst...")
            tf_decoder_dst = load_tf_pickle(decoder_dst_path)
            pt_decoder_dst = convert_decoder(tf_decoder_dst)

            out_path = model_dir / f"{dst_prefix}_decoder_dst.pth"
            torch.save(pt_decoder_dst, out_path)
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"\n[4/4] SKIP: {decoder_dst_path.name} not found")
    else:
        # ===== CONVERT INTER_B =====
        inter_b_path = model_dir / f"{src_prefix}_inter_B.npy"
        if inter_b_path.exists():
            print("\n[2/4] Converting inter_B...")
            tf_inter_b = load_tf_pickle(inter_b_path)
            pt_inter_b = convert_inter(tf_inter_b)

            out_path = model_dir / f"{dst_prefix}_inter_B.pth"
            torch.save(pt_inter_b, out_path)
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"[2/4] SKIP: {inter_b_path.name} not found")

        # ===== CONVERT INTER_AB (if exists) =====
        inter_ab_path = model_dir / f"{src_prefix}_inter_AB.npy"
        if inter_ab_path.exists():
            print("\n[3/4] Converting inter_AB...")
            tf_inter_ab = load_tf_pickle(inter_ab_path)
            pt_inter_ab = convert_inter(tf_inter_ab)

            out_path = model_dir / f"{dst_prefix}_inter_AB.pth"
            torch.save(pt_inter_ab, out_path)
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"\n[3/4] SKIP: {inter_ab_path.name} not found (normal for pretrain models)")
            print("  -> inter_AB will be randomly initialized (correct for pretrain->train transition)")

        # ===== CONVERT DECODER =====
        decoder_path = model_dir / f"{src_prefix}_decoder.npy"
        if decoder_path.exists():
            print("\n[4/4] Converting decoder...")
            tf_decoder = load_tf_pickle(decoder_path)
            pt_decoder = convert_decoder(tf_decoder)

            out_path = model_dir / f"{dst_prefix}_decoder.pth"
            torch.save(pt_decoder, out_path)
            print(f"  [OK] Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"\n[4/4] SKIP: {decoder_path.name} not found")
    
    # ===== CREATE SAENXT DATA FILE =====
    # The SAEHD _data.dat has different option key names than SAENXT.
    # We must translate SAEHD keys → SAENXT keys and fill in SAENXT defaults.
    data_dst = model_dir / f"{dst_prefix}_data.dat"
    if data_src.exists() and not data_dst.exists():
        saehd_data = pickle.loads(data_src.read_bytes())
        saehd_opts = saehd_data.get('options', {})
        
        # Map SAEHD keys → SAENXT keys
        nxt_opts = {
            # Carried over directly (same key name)
            'ae_dims':            saehd_opts.get('ae_dims', 512),
            'batch_size':         saehd_opts.get('batch_size', 8),
            'resolution':         saehd_opts.get('resolution', 224),
            'e_dims':             saehd_opts.get('e_dims', 64),
            'd_mask_dims':        saehd_opts.get('d_mask_dims', 32),
            'masked_training':    saehd_opts.get('masked_training', True),
            'eyes_mouth_prio':    saehd_opts.get('eyes_mouth_prio', False),
            'uniform_yaw':        saehd_opts.get('uniform_yaw', True),
            'random_warp':        saehd_opts.get('random_warp', True),
            'random_src_flip':    saehd_opts.get('random_src_flip', False),
            'random_dst_flip':    saehd_opts.get('random_dst_flip', False),
            'random_hsv_power':   0.0,  # SAEHD value causes cv2.merge crash in NCHW pipeline
            'gan_power':          saehd_opts.get('gan_power', 0.0),
            'target_iter':        saehd_opts.get('target_iter', 0),
            'blur_out_mask':      saehd_opts.get('blur_out_mask', False),
            'adabelief':          saehd_opts.get('adabelief', True),
            'ct_mode':            saehd_opts.get('ct_mode', 'none'),
            'face_type':          saehd_opts.get('face_type', 'whole_face'),
            
            # Renamed keys: SAEHD name → SAENXT name
            'decoder_dims':       saehd_opts.get('d_dims', 64),
            'gradient_clipping':  saehd_opts.get('clipgrad', True),
            'lr_dropout':         saehd_opts.get('lr_dropout', False),
            'autobackup_hour':    saehd_opts.get('autobackup_hour', 0),
            
            # SAENXT-only defaults (not in SAEHD)
            'archi':              dst_archi,
            'encoder_type':       'saenxt-v2',  # Sole active encoder; now restored to classic SAEHD stack
            'backbone_type':      'tiny',
            'learning_rate':      5e-5,
            'optimizer_type':     'adabelief' if saehd_opts.get('adabelief', True) else 'adam',
            'loss_function':      'SSIM',
            'mixed_precision':    'fp16',
            'background_blur':    True,
            'cuda_benchmark':     True,
            'maximum_n_backups':  24,
            'grad_checkpointing': False,
            'pretrain':           False,  # Always start as non-pretrain
            'growth_enabled':     False,
            'growth_stage':       0,
            'growth_base_resolution': saehd_opts.get('resolution', 224),
            'random_occlusion_prob': 0.0,
            'occlusion_images_path': '',
            'lpips_weight':       0.0,
            'vgg_weight':         0.0,
            'ffl_weight':         0.0,
            'l1_weight':          0.0,
            'landmark_weight':    0.0,
            'id_loss_weight':     0.0,
            'skin_detail_weight': 0.0,
            'edge_detail_weight': 0.0,
            'freq_band_weight':   0.0,
            'refiner_enabled':    False,
            'mask_encoder_input': False,
            'vram_debug':         False,
            'show_filenames':     False,
            'show_zoom_preview':  False,
            'show_error_heatmap_preview': False,
            'show_seam_preview':  False,
            'show_high_frequency_preview': False,
        }
        
        nxt_data = {
            'iter': saehd_data.get('iter', 0),
            'options': nxt_opts,
            'loss_history': saehd_data.get('loss_history', []),
            'sample_for_preview': saehd_data.get('sample_for_preview', None),
            'choosed_gpu_indexes': saehd_data.get('choosed_gpu_indexes', None),
        }
        
        data_dst.write_bytes(pickle.dumps(nxt_data))
        print(f"\n  [OK] Created {data_dst.name} with translated SAEHD options")
        print(f"       batch_size={nxt_opts['batch_size']}, resolution={nxt_opts['resolution']}, ae_dims={nxt_opts['ae_dims']}")
        print(f"       d_dims→decoder_dims={nxt_opts['decoder_dims']}, clipgrad→gradient_clipping={nxt_opts['gradient_clipping']}")
        print(f"       archi={nxt_opts['archi']}, encoder_type={nxt_opts['encoder_type']}, iter={nxt_data['iter']}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("  Conversion Complete!")
    print("=" * 70)
    if src_archi == 'df':
        converted_desc = (
            "  [OK] encoder     - SAEHD encoder (1:1 architecture port to SAEHDEncoder)\n"
            "  [OK] inter       - Dense bottleneck (TF->PyTorch format)\n"
            "  [OK] decoder_src - Source decoder (TF->PyTorch format)\n"
            "  [OK] decoder_dst - Destination decoder (TF->PyTorch format)\n"
        )
        skipped_desc = (
            "  [--] optimizer - Must restart fresh\n"
            "  [--] GAN       - Must restart fresh\n"
        )
        next_steps = (
            f"  1. Start SAENXT training with model name \"{dst_prefix}\"\n"
            "  2. archi='df-udt' and encoder_type='saenxt-v2' are stamped into the new SAENXT model\n"
            "  3. All pretrained weights will be loaded (encoder + inter + decoder_src + decoder_dst)\n"
            "  4. Review settings on first run, then continue training normally\n"
            "  5. Loss may jump initially, then recover - this is expected\n"
        )
    else:
        converted_desc = (
            "  [OK] encoder   - SAEHD encoder (1:1 architecture port to SAEHDEncoder)\n"
            "  [OK] inter_B   - Dense bottleneck (TF->PyTorch format)\n"
            "  [OK] decoder   - Upscale+Residual decoder (TF->PyTorch format)\n"
        )
        skipped_desc = (
            "  [--] inter_AB  - Not present / reset on pretrain->train\n"
            "  [--] optimizer - Must restart fresh\n"
            "  [--] GAN       - Must restart fresh\n"
        )
        next_steps = (
            f"  1. Start SAENXT training with model name \"{dst_prefix}\"\n"
            "  2. archi='liae-udt' and encoder_type='saenxt-v2' are stamped into the new SAENXT model\n"
            "  3. All pretrained weights will be loaded (encoder + decoder + inter_B)\n"
            "  4. Enable random_warp=True so Inter_AB can learn the encoder's features\n"
            "  5. Loss may jump initially, then recover - this is expected\n"
        )

    print(
        "\nWhat was converted:\n"
        f"{converted_desc}\n"
        "What was NOT converted (expected):\n"
        f"{skipped_desc}\n"
        "Next steps:\n"
        f"{next_steps}"
    )
    
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
