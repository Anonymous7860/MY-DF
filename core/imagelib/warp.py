import cv2
import numpy as np

from core import randomex


def gen_warp_params(
    w,
    flip=False,
    rotation_range=None,
    scale_range=None,
    tx_range=None,
    ty_range=None,
    rnd_state=None,
    warp_rnd_state=None,
):
    if ty_range is None:
        ty_range = [-0.05, 0.05]
    if tx_range is None:
        tx_range = [-0.05, 0.05]
    if scale_range is None:
        scale_range = [-0.5, 0.5]
    if rotation_range is None:
        rotation_range = [-10, 10]
    if rnd_state is None:
        rnd_state = np.random
    if warp_rnd_state is None:
        warp_rnd_state = np.random
    rw = None
    if w < 64:
        rw = w
        w = 64

    rotation = rnd_state.uniform(rotation_range[0], rotation_range[1])
    scale = rnd_state.uniform(1 + scale_range[0], 1 + scale_range[1])
    tx = rnd_state.uniform(tx_range[0], tx_range[1])
    ty = rnd_state.uniform(ty_range[0], ty_range[1])
    p_flip = flip and rnd_state.randint(10) < 4

    cell_size = [w // (2**i) for i in range(1, 4)][warp_rnd_state.randint(3)]
    cell_count = w // cell_size + 1
    # mapy is a transposed VIEW of mapx — mapx noise leaks into mapy before
    # mapy gets its own noise. This matches legacy DFL behavior and existing
    # trained models depend on this coupled warp distribution.
    grid_points = np.linspace(0, w, cell_count)
    mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
    mapy = mapx.T
    mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] + randomex.random_normal(
        size=(cell_count - 2, cell_count - 2), rnd_state=warp_rnd_state
    ) * (cell_size * 0.24)
    mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] + randomex.random_normal(
        size=(cell_count - 2, cell_count - 2), rnd_state=warp_rnd_state
    ) * (cell_size * 0.24)
    half_cell_size = cell_size // 2
    target_size = w + cell_size
    mapx = cv2.resize(mapx, (target_size, target_size))[
        half_cell_size:-half_cell_size, half_cell_size:-half_cell_size
    ].astype(np.float32)
    mapy = cv2.resize(mapy, (target_size, target_size))[
        half_cell_size:-half_cell_size, half_cell_size:-half_cell_size
    ].astype(np.float32)

    random_transform_mat = cv2.getRotationMatrix2D((int(w / 2), int(w / 2)), rotation, scale)
    random_transform_mat[:, 2] += (tx * w, ty * w)

    u_mat = random_transform_mat.copy()
    u_mat[:, 2] /= w

    return {
        "mapx": mapx,
        "mapy": mapy,
        "rmat": random_transform_mat,
        "umat": u_mat,
        "w": w,
        "rw": rw,
        "flip": p_flip,
    }


def _single_pass_warp_map(params, apply_flip):
    """Compose DFL's deformation and affine maps for one image resample."""
    coordinate_map = np.dstack((params["mapx"], params["mapy"]))
    composed_map = cv2.warpAffine(
        coordinate_map,
        params["rmat"],
        (params["w"], params["w"]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32, copy=False)
    if apply_flip:
        composed_map = composed_map[:, ::-1, :]
    return np.ascontiguousarray(composed_map)


def warp_by_params(
    params,
    img,
    can_warp,
    can_transform,
    can_flip,
    border_replicate,
    cv2_inter=cv2.INTER_CUBIC,
    single_pass=False,
):
    rw = params["rw"]

    # The fused path is intentionally narrow. It is equivalent to the normal
    # FACE_IMAGE sequence (deformation -> affine transform -> optional flip),
    # while sampling RGB only once. Other border modes, partial transforms,
    # masks, and the legacy sub-64px resize path retain the original behavior.
    use_single_pass = bool(
        single_pass
        and can_warp
        and can_transform
        and border_replicate
        and rw is None
    )
    if use_single_pass:
        apply_flip = bool(can_flip and params["flip"])
        composed_map = _single_pass_warp_map(params, apply_flip)
        img = cv2.remap(
            img,
            composed_map[..., 0],
            composed_map[..., 1],
            interpolation=cv2_inter,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if len(img.shape) == 2:
            img = img[..., None]
        return img

    if (can_warp or can_transform) and rw is not None:
        img = cv2.resize(img, (64, 64), interpolation=cv2_inter)

    if can_warp:
        img = cv2.remap(img, params["mapx"], params["mapy"], cv2_inter)
    if can_transform:
        img = cv2.warpAffine(
            img,
            params["rmat"],
            (params["w"], params["w"]),
            borderMode=(cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT),
            flags=cv2_inter,
        )

    if (can_warp or can_transform) and rw is not None:
        img = cv2.resize(img, (rw, rw), interpolation=cv2_inter)

    if len(img.shape) == 2:
        img = img[..., None]
    if can_flip and params["flip"]:
        img = img[:, ::-1, ...]
    return img
