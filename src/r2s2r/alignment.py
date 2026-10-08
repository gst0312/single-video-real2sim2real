"""The similarity fit both alignment steps are built on.

`align_scene_to_base.py` uses it to put the reconstruction in the robot base frame from
ArUco corner correspondences, and `solve_capture_qpos.py` uses it inside the iterative
closest point loop that refines that frame against the arm the reconstruction contains.
"""

import numpy as np


def umeyama(src, dst, with_scale=True):
    """Similarity transform mapping src onto dst (Umeyama 1991).

    Returns (scale, rotation, translation) with dst ~= scale * rotation @ src + translation.
    With `with_scale` False the scale is held at one, which is the rigid Procrustes fit.
    """
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    if with_scale:
        var = (src_c**2).sum() / len(src)
        scale = float(np.trace(np.diag(d) @ s) / var)
    else:
        scale = 1.0
    trans = dst_mean - scale * rot @ src_mean
    return scale, rot, trans
