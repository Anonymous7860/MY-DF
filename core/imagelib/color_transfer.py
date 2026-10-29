import cv2
import numexpr as ne
import numpy as np
from numpy import linalg as npla
import random
from scipy.stats import special_ortho_group
import scipy as sp
from numpy import linalg as npla


def color_transfer_sot(src,trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer
    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter
    return value - clip it manually
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src value must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg value must be float")

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    src_dtype = src.dtype
    h,w,c = src.shape
    new_src = src.copy()

    advect = np.empty ( (h*w,c), dtype=src_dtype )
    for step in range (steps):
        advect.fill(0)
        for batch in range (batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= npla.norm(dir)

            projsource = np.sum( new_src*dir, axis=-1).reshape ((h*w))
            projtarget = np.sum( trg*dir, axis=-1).reshape ((h*w))

            idSource = np.argsort (projsource)
            idTarget = np.argsort (projtarget)

            a = projtarget[idTarget]-projsource[idSource]
            for i_c in range(c):
                advect[idSource,i_c] += a * dir[i_c]
        new_src += advect.reshape( (h,w,c) ) / batch_size

    if reg_sigmaXY != 0.0:
        src_diff = new_src-src
        src_diff_filt = cv2.bilateralFilter (src_diff, 0, reg_sigmaV, reg_sigmaXY )
        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[...,None]
        new_src = src + src_diff_filt
    return new_src

def color_transfer_mkl(x0, x1):
    eps = np.finfo(float).eps

    h,w,c = x0.shape
    h1,w1,c1 = x1.shape

    x0 = x0.reshape ( (h*w,c) )
    x1 = x1.reshape ( (h1*w1,c1) )

    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None)))

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

    Da_inv = np.diag(1./(np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)

    result = np.dot(x0-mx0, t) + mx1
    return np.clip ( result.reshape ( (h,w,c) ).astype(x0.dtype), 0, 1)

def color_transfer_idt(i0, i1, bins=256, n_rot=20):
    import scipy.stats
    
    relaxation = 1 / n_rot
    h,w,c = i0.shape
    h1,w1,c1 = i1.shape

    i0 = i0.reshape ( (h*w,c) )
    i1 = i1.reshape ( (h1*w1,c1) )

    n_dims = c

    d0 = i0.T
    d1 = i1.T

    for i in range(n_rot):

        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)

        for j in range(n_dims):

            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _     = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

    return np.clip ( d0.T.reshape ( (h,w,c) ).astype(i0.dtype) , 0, 1)

def reinhard_color_transfer(target : np.ndarray, source : np.ndarray, target_mask : np.ndarray = None, source_mask : np.ndarray = None, mask_cutoff=0.5) -> np.ndarray:
    """
    Transfer color using rct method.

        target      np.ndarray H W 3C   (BGR)   np.float32
        source      np.ndarray H W 3C   (BGR)   np.float32

        target_mask(None)   np.ndarray H W 1C  np.float32
        source_mask(None)   np.ndarray H W 1C  np.float32
        
        mask_cutoff(0.5)    float

    masks are used to limit the space where color statistics will be computed to adjust the target

    reference: Color Transfer between Images https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    source_input = source
    if source_mask is not None:
        source_input = source_input.copy()
        source_input[source_mask[...,0] < mask_cutoff] = [0,0,0]
    
    target_input = target
    if target_mask is not None:
        target_input = target_input.copy()
        target_input[target_mask[...,0] < mask_cutoff] = [0,0,0]

    target_l_mean, target_l_std, target_a_mean, target_a_std, target_b_mean, target_b_std, \
        = target_input[...,0].mean(), target_input[...,0].std(), target_input[...,1].mean(), target_input[...,1].std(), target_input[...,2].mean(), target_input[...,2].std()
    
    source_l_mean, source_l_std, source_a_mean, source_a_std, source_b_mean, source_b_std, \
        = source_input[...,0].mean(), source_input[...,0].std(), source_input[...,1].mean(), source_input[...,1].std(), source_input[...,2].mean(), source_input[...,2].std()
    
    # not as in the paper: scale by the standard deviations using reciprocal of paper proposed factor
    target_l = target[...,0]
    target_l = ne.evaluate('(target_l - target_l_mean) * source_l_std / target_l_std + source_l_mean')

    target_a = target[...,1]
    target_a = ne.evaluate('(target_a - target_a_mean) * source_a_std / target_a_std + source_a_mean')
    
    target_b = target[...,2]
    target_b = ne.evaluate('(target_b - target_b_mean) * source_b_std / target_b_std + source_b_mean')

    np.clip(target_l,    0, 100, out=target_l)
    np.clip(target_a, -127, 127, out=target_a)
    np.clip(target_b, -127, 127, out=target_b)

    return cv2.cvtColor(np.stack([target_l,target_a,target_b], -1), cv2.COLOR_LAB2BGR)


def linear_color_transfer(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape( t.shape[-1],-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2,0,1).reshape( s.shape[-1],-1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    return np.clip(matched_img.astype(source_img.dtype), 0, 1)

def lab_image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)

    mn = arr.min()
    mx = arr.max()
    scale_range = (max([mn, 0]), min([mx, 255]))

    if mn < scale_range[0] or mx > scale_range[1]:
        return (scale_range[1] - scale_range[0]) * (arr - mn) / (mx - mn) + scale_range[0]

    return arr

def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def color_hist_match(src_im, tar_im, hist_match_threshold=255):
    h,w,c = src_im.shape
    matched_R = channel_hist_match(src_im[:,:,0], tar_im[:,:,0], hist_match_threshold, None)
    matched_G = channel_hist_match(src_im[:,:,1], tar_im[:,:,1], hist_match_threshold, None)
    matched_B = channel_hist_match(src_im[:,:,2], tar_im[:,:,2], hist_match_threshold, None)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += ( src_im[:,:,i],)


    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched

def color_transfer_mix(img_src,img_trg):
    img_src = np.clip(img_src*255.0, 0, 255).astype(np.uint8)
    img_trg = np.clip(img_trg*255.0, 0, 255).astype(np.uint8)

    img_src_lab = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    img_trg_lab = cv2.cvtColor(img_trg, cv2.COLOR_BGR2LAB)

    rct_light = np.clip ( linear_color_transfer(img_src_lab[...,0:1].astype(np.float32)/255.0,
                                                img_trg_lab[...,0:1].astype(np.float32)/255.0 )[...,0]*255.0,
                          0, 255).astype(np.uint8)

    img_src_lab[...,0] = (np.ones_like (rct_light)*100).astype(np.uint8)
    img_src_lab = cv2.cvtColor(img_src_lab, cv2.COLOR_LAB2BGR)

    img_trg_lab[...,0] = (np.ones_like (rct_light)*100).astype(np.uint8)
    img_trg_lab = cv2.cvtColor(img_trg_lab, cv2.COLOR_LAB2BGR)

    img_rct = color_transfer_sot( img_src_lab.astype(np.float32), img_trg_lab.astype(np.float32) )
    img_rct = np.clip(img_rct, 0, 255).astype(np.uint8)

    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_BGR2LAB)
    img_rct[...,0] = rct_light
    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_LAB2BGR)


    return (img_rct / 255.0).astype(np.float32)

def color_transfer(ct_mode, img_src, img_trg, seed=None, exclusion_mask=None):
    """
    color transfer for [0,1] float32 inputs
    """
    if ct_mode == 'lct':
        out = linear_color_transfer (img_src, img_trg)
    elif ct_mode == 'rct':
        out = reinhard_color_transfer(img_src, img_trg)
    elif ct_mode == 'mkl':
        out = color_transfer_mkl (img_src, img_trg)
    elif ct_mode == 'idt':
        out = color_transfer_idt (img_src, img_trg)
    elif ct_mode == 'sot':
        out = color_transfer_sot (img_src, img_trg)
        out = np.clip( out, 0.0, 1.0)
    elif ct_mode == 'mix-m':
        out = color_transfer_mix(img_src, img_trg)
    elif ct_mode == 'd-aug-src':
        out = dynamic_augmentation_src(img_src, seed=seed, exclusion_mask=exclusion_mask)
    elif ct_mode == 'd-aug-dst':
        out = dynamic_augmentation_dst(img_src, seed=seed)
    elif ct_mode == 'fs-aug':
        out = color_augmentation(img_src, seed=seed)
    elif ct_mode == 'cc-aug':
        out = cc_aug(img_src, seed=seed)
    elif ct_mode in ('d-aug-lite-src', 'prism-aug-src'):
        out = d_aug_lite_src(img_src, seed=seed, exclusion_mask=exclusion_mask)
    elif ct_mode in ('d-aug-lite-dst', 'prism-aug-dst'):
        out = d_aug_lite_dst(img_src, seed=seed)
    else:
        raise ValueError(f"unknown ct_mode {ct_mode}")
    return out


# imported from faceswap
def color_augmentation(img, seed=None):
    """ Color adjust RGB image """
    img = img.astype(np.float32)
    face = img
    face = np.clip(face*255.0, 0, 255).astype(np.uint8)
    face = random_clahe(face, seed)
    face = random_lab(face, seed)
    img[:, :, :3] = face
    return (face / 255.0).astype(np.float32)

def random_lab_rotation(image, seed=None):
    """
    Randomly rotates image color around the L axis in LAB colorspace,
    keeping perceptual lightness constant.
    """
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2LAB)
    M = np.eye(3)
    M[1:, 1:] = special_ortho_group.rvs(2, 1, seed)
    image = image.dot(M)
    l, a, b = cv2.split(image)
    l = np.clip(l, 0, 100)
    a = np.clip(a, -127, 127)
    b = np.clip(b, -127, 127)
    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_LAB2BGR)
    np.clip(image, 0, 1, out=image)
    return image

def random_lab(image, seed=None):
    """ Perform random color/lightness adjustment in L*a*b* colorspace """
    random.seed(seed)
    amount_l = 30 / 100
    amount_ab = 8 / 100
    randoms = [(random.random() * amount_l * 2) - amount_l,  # L adjust
                (random.random() * amount_ab * 2) - amount_ab,  # A adjust
                (random.random() * amount_ab * 2) - amount_ab]  # B adjust
    image = cv2.cvtColor(  # pylint:disable=no-member
    image, cv2.COLOR_BGR2LAB).astype("float32") / 255.0  # pylint:disable=no-member

    for idx, adjustment in enumerate(randoms):
        if adjustment >= 0:
            image[:, :, idx] = ((1 - image[:, :, idx]) * adjustment) + image[:, :, idx]
        else:
            image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
    image = cv2.cvtColor((image * 255.0).astype("uint8"),  # pylint:disable=no-member
                        cv2.COLOR_LAB2BGR)  # pylint:disable=no-member
    return image

def random_clahe(image, seed=None):
    """ Randomly perform Contrast Limited Adaptive Histogram Equalization """
    random.seed(seed)
    contrast_random = random.random()
    if contrast_random > 50 / 100:
        return image

    base_contrast = 1 # testing because it breaks on small sizes
    grid_base = random.random() * 4
    contrast_adjustment = int(grid_base * (base_contrast / 2))
    grid_size = base_contrast + contrast_adjustment

    clahe = cv2.createCLAHE(clipLimit=2.0,  # pylint: disable=no-member
                            tileGridSize=(grid_size, grid_size))
    for chan in range(3):
        image[:, :, chan] = clahe.apply(image[:, :, chan])
    return image

def cc_random_lab(image, seed=None):
    """ Perform random color/lightness adjustment in L*a*b* colorspace (increased aug)"""
    random.seed(seed)
    amount_l = 45 / 100
    amount_ab = 20 / 100
    randoms = [(random.random() * amount_l * 2) - amount_l,  # L adjust
                (random.random() * amount_ab * 2) - amount_ab,  # A adjust
                (random.random() * amount_ab * 2) - amount_ab]  # B adjust
    image = cv2.cvtColor(
    image, cv2.COLOR_BGR2LAB).astype("float32") / 255.0

    for idx, adjustment in enumerate(randoms):
        if adjustment >= 0:
            image[:, :, idx] = ((1 - image[:, :, idx]) * adjustment) + image[:, :, idx]
        else:
            image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
    image = cv2.cvtColor((image * 255.0).astype("uint8"),
                        cv2.COLOR_LAB2BGR)
    return image

def cc_aug(img, seed=None):
    """Color adjust RGB image with increased augmentation range"""
    img = img.astype(np.float32)
    face = img
    face = np.clip(face * 255.0, 0, 255).astype(np.uint8)
    
    face = random_clahe(face, seed)
    face = cc_random_lab(face, seed)
    
    brightness_factor_range = (0.7, 1.1)
    contrast_factor_range = (0.7, 1.1)
    color_factor_range = (0.6, 1.2)
    
    face = adjust_brightness(face, brightness_factor_range, seed)
    face = adjust_contrast(face, contrast_factor_range, seed)
    face = adjust_color(face, color_factor_range, seed)
    
    img[:, :, :3] = face
    return (face / 255.0).astype(np.float32)

def random_color_temperature(image, seed=None):
    """Simulates a random color temperature and tint shift."""
    rs = np.random.RandomState(seed)
    image = image.astype(np.float32)
    temp_shift = rs.uniform(-20, 20)
    image[..., 2] = np.clip(image[..., 2] + temp_shift, 0, 255)
    image[..., 0] = np.clip(image[..., 0] - temp_shift, 0, 255)
    tint_shift = rs.uniform(-10, 10)
    image[..., 1] = np.clip(image[..., 1] + tint_shift, 0, 255)
    return image.astype(np.uint8)

def _random_vignette(image, rs):
    """Adds a random vignette effect to the image."""
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, rs.uniform(50, 150))
    kernel_y = cv2.getGaussianKernel(rows, rs.uniform(50, 150))
    kernel = kernel_y * kernel_x.T
    if rs.rand() > 0.5:
        kernel = 1 - kernel
    mask = kernel / np.linalg.norm(kernel)
    mask = mask / mask.max()
    vignette_intensity = rs.uniform(0.5, 0.9)
    if rs.rand() < 0.25:
        color = rs.rand(3) * 0.3
        for i in range(3):
            image[..., i] = image[..., i] * (1 - (1 - mask) * vignette_intensity) + (255 * color[i] * (1-mask) * vignette_intensity)
    else:
         for i in range(3):
            image[..., i] = image[..., i] * mask
    return np.clip(image, 0, 255).astype(np.uint8)

def _random_channel_shift(image, rs, max_shift=4):
    """Simulates chromatic aberration by slightly shifting color channels."""
    image = image.astype(np.float32)
    r, g, b = cv2.split(image)
    r_shift_x, r_shift_y = rs.randint(-max_shift, max_shift + 1, size=2)
    b_shift_x, b_shift_y = rs.randint(-max_shift, max_shift + 1, size=2)
    r_M = np.float32([[1, 0, r_shift_x], [0, 1, r_shift_y]])
    b_M = np.float32([[1, 0, b_shift_x], [0, 1, b_shift_y]])
    r = cv2.warpAffine(r, r_M, (image.shape[1], image.shape[0]))
    b = cv2.warpAffine(b, b_M, (image.shape[1], image.shape[0]))
    shifted_image = cv2.merge([r, g, b])
    return np.clip(shifted_image, 0, 255).astype(np.uint8)

def _adjust_exposure(image, rs):
    """Simulates random over/underexposure using a gamma curve."""
    gamma = rs.uniform(0.7, 1.5)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def _adjust_exposure_range(image, rs, gamma_range=(0.88, 1.16)):
    """A tighter exposure curve for gentler illumination robustness."""
    gamma = rs.uniform(*gamma_range)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def _random_film_grain(image, rs):
    """Adds realistic film grain."""
    h, w, c = image.shape
    noise = rs.normal(0, rs.uniform(2, 6), (h, w, c))
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _random_gradient_overlay(image, rs):
    """Overlays a random colored gradient to simulate environmental lighting."""
    h, w, _ = image.shape
    x1, y1 = rs.randint(0, w), rs.randint(0, h)
    x2, y2 = rs.randint(-w//2, w + w//2), rs.randint(-h//2, h + h//2)
    color = rs.rand(3)
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return image.astype(np.uint8)
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    denominator = dx**2 + dy**2
    if denominator == 0:
        return image.astype(np.uint8)
    numerator = (x_coords - x1) * dx + (y_coords - y1) * dy
    t = numerator / denominator
    t = np.clip(t, 0, 1)
    gradient = t[..., np.newaxis] * color
    blend_mode = rs.choice(['overlay', 'soft_light', 'add'])
    intensity = rs.uniform(0.2, 0.6)
    img_f = image.astype(np.float32) / 255.0
    if blend_mode == 'overlay':
        low_mask = img_f <= 0.5
        high_mask = ~low_mask
        img_f_overlay = np.zeros_like(img_f)
        img_f_overlay[low_mask] = 2 * img_f[low_mask] * gradient[low_mask]
        img_f_overlay[high_mask] = 1 - 2 * (1 - img_f[high_mask]) * (1 - gradient[high_mask])
        img_f = img_f_overlay
    elif blend_mode == 'soft_light':
        img_f = (1 - 2 * gradient) * img_f**2 + 2 * gradient * img_f
    elif blend_mode == 'add':
        img_f += gradient * intensity
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)

def _mild_color_temperature(image, rs, temp_range=(-10, 10), tint_range=(-5, 5)):
    """A gentler temperature/tint shift than the heavy d-aug version."""
    image = image.astype(np.float32)
    temp_shift = rs.uniform(*temp_range)
    tint_shift = rs.uniform(*tint_range)
    image[..., 2] = np.clip(image[..., 2] + temp_shift, 0, 255)
    image[..., 0] = np.clip(image[..., 0] - temp_shift, 0, 255)
    image[..., 1] = np.clip(image[..., 1] + tint_shift, 0, 255)
    return image.astype(np.uint8)

def _soft_gradient_overlay(image, rs, intensity_range=(0.08, 0.22)):
    """A low-amplitude environmental lighting wash for mild domain robustness."""
    h, w, _ = image.shape
    x1, y1 = rs.randint(0, w), rs.randint(0, h)
    x2, y2 = rs.randint(-w // 3, w + w // 3), rs.randint(-h // 3, h + h // 3)
    color = rs.rand(3)
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return image.astype(np.uint8)

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    denominator = dx**2 + dy**2
    if denominator == 0:
        return image.astype(np.uint8)

    numerator = (x_coords - x1) * dx + (y_coords - y1) * dy
    t = np.clip(numerator / denominator, 0, 1)[..., np.newaxis]
    gradient = t * color

    img_f = image.astype(np.float32) / 255.0
    blend_mode = rs.choice(['soft_light', 'add'])
    intensity = rs.uniform(*intensity_range)

    if blend_mode == 'soft_light':
        mixed = (1 - 2 * gradient) * img_f**2 + 2 * gradient * img_f
        img_f = img_f * (1.0 - intensity) + mixed * intensity
    else:
        img_f = np.clip(img_f + gradient * intensity, 0.0, 1.0)

    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

def adjust_brightness(image, factor_range, seed=None):
    """Adjusts the brightness of an image."""
    np.random.seed(seed)
    brightness_factor = np.random.uniform(*factor_range)
    return np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

def adjust_contrast(image, factor_range, seed=None):
    """Adjusts the contrast of an image."""
    np.random.seed(seed)
    mean_value = np.mean(image, axis=(0, 1), keepdims=True)
    contrast_factor = np.random.uniform(*factor_range)
    return np.clip((image - mean_value) * contrast_factor + mean_value, 0, 255).astype(np.uint8)

def adjust_color(image, factor_range, seed=None):
    """Adjusts the color of an image."""
    np.random.seed(seed)
    color_factor = np.random.uniform(*factor_range)
    return np.clip(image * color_factor, 0, 255).astype(np.uint8)

def _execute_aug_pipeline(img, augmentations, seed=None):
    """Helper to execute an augmentation pipeline."""
    original_shape = img.shape
    is_nchw = len(original_shape) == 3 and original_shape[0] < original_shape[1] and original_shape[0] < original_shape[2]
    if is_nchw:
        img = np.transpose(img, (1, 2, 0))
    face = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    rs = np.random.RandomState(seed)
    rs.shuffle(augmentations)
    for aug_func, probability in augmentations:
        if rs.rand() < probability:
            result = aug_func(face, rs=rs)
            if result.dtype == np.float32:
                face = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            else:
                face = result
    result_img = (face / 255.0).astype(np.float32)
    if is_nchw:
        result_img = np.transpose(result_img, (2, 0, 1))
    return result_img

def dynamic_augmentation_src(img, seed=None, exclusion_mask=None):
    """
    A 'lite' version of d-aug for SRC, focusing on color and lighting without heavy distortions.
    """
    from .shadows import shadow_highlights_augmentation
    if exclusion_mask is not None:
        original_img = img.copy()
    augmentations = [
        (lambda x, rs: cc_random_lab(x, seed=rs.randint(0x80000000)), 0.8),
        (lambda x, rs: _adjust_exposure(x, rs), 0.6),
        (lambda x, rs: adjust_brightness(x, factor_range=(0.75, 1.25), seed=rs.randint(0x80000000)), 0.5),
        (lambda x, rs: adjust_color(x, factor_range=(0.8, 1.2), seed=rs.randint(0x80000000)), 0.4),
        (lambda x, rs: random_color_temperature(x, seed=rs.randint(0x80000000)), 0.5),
        (lambda x, rs: adjust_contrast(x, factor_range=(0.85, 1.15), seed=rs.randint(0x80000000)), 0.6),
        (lambda x, rs: shadow_highlights_augmentation(x.astype(np.float32)/255.0, high_ratio=(rs.uniform(0.9, 1.2), rs.uniform(1.2, 2.0)), low_ratio=(rs.uniform(0.2, 0.4), rs.uniform(0.4, 0.6)), seed=rs.randint(0x80000000)), 0.4),
        (lambda x, rs: _random_gradient_overlay(x, rs), 0.4),
        (lambda x, rs: _random_channel_shift(x, rs, max_shift=2), 0.3),
    ]
    augmented_img = _execute_aug_pipeline(img, augmentations, seed)
    if exclusion_mask is not None:
        if exclusion_mask.shape[-1] != augmented_img.shape[-1]:
             exclusion_mask = np.repeat(exclusion_mask, augmented_img.shape[-1], axis=-1)
        return augmented_img * (1 - exclusion_mask) + original_img * exclusion_mask
    else:
        return augmented_img

def dynamic_augmentation_dst(img, seed=None):
    """
    The "full everything" d-aug for DST, with aggressive augmentations.
    """
    from .shadows import shadow_highlights_augmentation
    augmentations = [
        (lambda x, rs: shadow_highlights_augmentation(x.astype(np.float32)/255.0, high_ratio=(rs.uniform(0.8, 1.8), rs.uniform(1.8, 3.8)), low_ratio=(rs.uniform(0.05, 0.25), rs.uniform(0.25, 0.55)), seed=rs.randint(0x80000000)), 0.6),
        (lambda x, rs: _adjust_exposure(x, rs), 0.6),
        (lambda x, rs: adjust_brightness(x, factor_range=(0.55, 1.45), seed=rs.randint(0x80000000)), 0.6),
        (lambda x, rs: _random_gradient_overlay(x, rs), 0.5),
        (lambda x, rs: cc_random_lab(x, seed=rs.randint(0x80000000)), 0.75),
        (lambda x, rs: random_color_temperature(x, seed=rs.randint(0x80000000)), 0.5),
        (lambda x, rs: adjust_color(x, factor_range=(0.65, 1.35), seed=rs.randint(0x80000000)), 0.5),
        (lambda x, rs: random_lab_rotation(x.astype(np.float32)/255.0, seed=rs.randint(0x80000000)), 0.3),
        (lambda x, rs: adjust_contrast(x, factor_range=(0.6, 1.4), seed=rs.randint(0x80000000)), 0.8),
        (lambda x, rs: random_clahe(x, seed=rs.randint(0x80000000)), 0.35),
        (lambda x, rs: _random_vignette(x, rs), 0.5),
        (lambda x, rs: _random_channel_shift(x, rs, max_shift=6), 0.4),
        (lambda x, rs: _random_film_grain(x, rs), 0.5),
    ]
    return _execute_aug_pipeline(img, augmentations, seed)

def d_aug_lite_src(img, seed=None, exclusion_mask=None):
    """
    D-Aug Lite SRC branch:
    keep the proven fs-aug behavior as the base so SRC identity/color fidelity stays stable.
    """
    if exclusion_mask is not None:
        original_img = img.copy()

    augmented_img = color_augmentation(img, seed=seed)
    if exclusion_mask is not None:
        if exclusion_mask.shape[-1] != augmented_img.shape[-1]:
            exclusion_mask = np.repeat(exclusion_mask, augmented_img.shape[-1], axis=-1)
        return augmented_img * (1 - exclusion_mask) + original_img * exclusion_mask
    return augmented_img

def d_aug_lite_dst(img, seed=None):
    """
    D-Aug Lite DST branch:
    start from fs-aug, then add only a little extra illumination/domain pressure.
    """
    from .shadows import shadow_highlights_augmentation

    base = color_augmentation(img, seed=seed)
    augmentations = [
        (lambda x, rs: _adjust_exposure_range(x, rs, gamma_range=(0.97, 1.05)), 0.18),
        (lambda x, rs: _mild_color_temperature(x, rs, temp_range=(-4, 4), tint_range=(-2, 2)), 0.16),
        (
            lambda x, rs: shadow_highlights_augmentation(
                x.astype(np.float32) / 255.0,
                high_ratio=(rs.uniform(1.00, 1.04), rs.uniform(1.04, 1.10)),
                low_ratio=(rs.uniform(0.78, 0.84), rs.uniform(0.84, 0.94)),
                seed=rs.randint(0x80000000),
            ),
            0.08,
        ),
        (lambda x, rs: _soft_gradient_overlay(x, rs, intensity_range=(0.04, 0.08)), 0.05),
    ]
    return _execute_aug_pipeline(base, augmentations, None if seed is None else seed + 1337)


# Backward-compatible direct-call aliases for configurations and extensions
# created while this mode was named Prism.
prism_augmentation_src = d_aug_lite_src
prism_augmentation_dst = d_aug_lite_dst
