"""粗定向的桥：用图像互相关把「大致对着某个方向（±10°）」收敛到 0.3° 以内。

为什么需要它
  像素级点匹配的有效半径只有几像素（165″/px ⇒ 6 px ≈ 0.28°）。要在 ±10° 上做
  3 自由度网格，步长必须 ≤0.15°，那是 200 万次评估，做不到。
  但「把星表渲染成图，和实拍图做互相关」对位移是不敏感的：一次 FFT 就能把
  0–10° 的平移一次找出来，剩下的只有滚转一个自由度量，21 次 FFT 就够。

流程
  1. 对每个候选滚转 θ：M = M0 @ Rz(θ)，把亮星渲染成图，与实拍星点图做 FFT 互相关；
     取峰值高度作为 θ 的得分（对齐时峰值高一个数量级）。
  2. 最好 θ 的峰值位置给出 (dx, dy)，用数值雅可比把它翻译成一个小的姿态修正。
  3. 交回 attitude.calibrate_frame 从 15 px 半径接管，精修到 0.16 px。
"""
import numpy as np

import config as CF
from src import attitude


def splat(u, v, w, h, bin_size=1):
    """把点云累加到 (h, w) 的图像上（按 bin_size 粗化）。"""
    img = np.zeros((h, w), np.float32)
    if len(u) == 0:
        return img
    ix = (u / bin_size).astype(np.int64)
    iy = (v / bin_size).astype(np.int64)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    np.add.at(img, (iy[ok], ix[ok]), 1.0)
    return img


def project(C, M, w, h):
    cc = C @ M
    z = cc[:, 2]
    ok = z > 0.15
    u = CF.FOC * cc[:, 0] / np.maximum(z, 1e-9) + CF.CX
    v = CF.FOC * cc[:, 1] / np.maximum(z, 1e-9) + CF.CY
    ins = ok & (u > 0) & (u < w) & (v > 0) & (v < h)
    return u[ins], v[ins]


def corr_peak(A, B):
    """A、B 同尺寸。返回 (峰值, dx, dy)，满足 A 平移 (dx,dy) 后与 B 最重合。"""
    FA = np.fft.rfft2(A - A.mean())
    FB = np.fft.rfft2(B - B.mean())
    cc = np.fft.irfft2(FA * np.conj(FB), s=A.shape)
    i = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = i
    if dy > A.shape[0] // 2:
        dy -= A.shape[0]
    if dx > A.shape[1] // 2:
        dx -= A.shape[1]
    return float(cc[i]), float(dx), float(dy)


def coarse_lock(M0, det_uv, C, roll_span=20.0, roll_step=2.0, verbose=True):
    """已知「大致」姿态 M0，滚转扫描 + 互相关，逐级细化到 ~0.2°。

    两级：4× 降采样（覆盖 ±45° 的平移）→ 原分辨率（0.02°/bin）。
    """
    M = M0
    diag = {}
    for bin_size, rspan, rstep in ((4, roll_span, roll_step), (1, 4.0, 0.25)):
        w = CF.IMG_W // bin_size
        h = CF.IMG_H // bin_size
        ox = splat(det_uv[:, 0], det_uv[:, 1], w, h, bin_size)
        best = (-1.0, 0.0, 0.0, 0.0)
        for th in np.arange(-rspan, rspan + 1e-9, rstep):
            Mt = M @ attitude.rotz(np.radians(th))
            u, v = project(C, Mt, CF.IMG_W, CF.IMG_H)
            pk, dx, dy = corr_peak(splat(u, v, w, h, bin_size), ox)
            if pk > best[0]:
                best = (pk, float(th), dx, dy)
        pk, th, dx, dy = best
        M = M @ attitude.rotz(np.radians(th))
        M = apply_pixel_shift(M, -dx * bin_size, -dy * bin_size)
        diag = dict(roll=th, dx=-dx * bin_size, dy=-dy * bin_size, peak=pk)
        if verbose:
            print(f'  互相关 bin={bin_size}: 滚转 {th:+.2f}°  平移 '
                  f'({-dx * bin_size:+.1f},{-dy * bin_size:+.1f})px  峰值 {pk:.0f}')
    return M, diag


def shift_to_rot(dx, dy):
    """图像整体平移 (dx,dy) 像素 对应的小旋转（相机系轴角）。

    由 c' = (I - [ω]×) c 在 c=e3 处得 du = -FOC·ω_y、dv = +FOC·ω_x，
    所以 ω = (dy/FOC, -dx/FOC, 0)。中心处精确，边缘是小角近似。
    """
    return np.array([dy / CF.FOC, -dx / CF.FOC, 0.0])


def apply_pixel_shift(M, dx, dy):
    """把姿态改成「投影整体平移 (dx,dy) 像素」。"""
    return M @ attitude.expm_rot(shift_to_rot(dx, dy))
