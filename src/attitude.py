"""逐帧绝对姿态标定：相机系三轴小旋转 + 径向尺度，对星表做 Gauss-Newton。

用法：
    acc, scale, nin, med = calib_frame_scale(M0, cat_dirs, G, detections)
    M = M0 @ acc                      # M0 是粗姿态
"""
import numpy as np

import config as CF
from src import prep


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expm_rot(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    K = skew(v / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def project_raw(C, M, foc=None):
    """星表方向 -> 像素坐标（不过滤），返回 (uv, ok)。"""
    foc = CF.FOC if foc is None else foc
    cc = C @ M
    z = cc[:, 2]
    ok = z > 0.12
    u = foc * cc[:, 0] / np.maximum(z, 1e-9) + CF.CX
    v = foc * cc[:, 1] / np.maximum(z, 1e-9) + CF.CY
    return np.stack([u, v], 1), ok


def match_hash(pred, det, rad, nb=10):
    """网格哈希最近邻。返回 (pred 行号, det 行号, 距离)。"""
    bk = {}
    for i, (x, y) in enumerate(det):
        bk.setdefault((int(x // nb), int(y // nb)), []).append(i)
    pi, di, dd = [], [], []
    for j, (x, y) in enumerate(pred):
        cx, cy = int(x // nb), int(y // nb)
        best, bi = rad * rad, -1
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                for i in bk.get((cx + ax, cy + ay), ()):
                    d2 = (det[i, 0] - x) ** 2 + (det[i, 1] - y) ** 2
                    if d2 < best:
                        best, bi = d2, i
        if bi >= 0:
            pi.append(j)
            di.append(bi)
            dd.append(np.sqrt(best))
    return np.array(pi), np.array(di), np.array(dd)


def calib_frame_scale(M0, cat_dirs, G, det, gmax=9.0, iters=10):
    """返回 (acc, scale, 内点数, 中位残差px)。acc 为相机系修正旋转。"""
    Cc = cat_dirs[G < gmax]
    acc = np.eye(3)
    sc = 1.0
    radii = [15.0, 6.0, 2.5, 1.2, 0.8, 0.6, 0.5, 0.4, 0.4, 0.4]
    for it in range(iters):
        M = M0 @ acc
        predF, okF = project_raw(Cc, M)
        pred0 = np.stack([CF.CX + sc * (predF[:, 0] - CF.CX),
                          CF.CY + sc * (predF[:, 1] - CF.CY)], 1)
        ins = okF & (pred0[:, 0] > 0) & (pred0[:, 0] < CF.IMG_W) \
                  & (pred0[:, 1] > 0) & (pred0[:, 1] < CF.IMG_H)
        rad = radii[min(it, len(radii) - 1)]
        pi, di, dd = match_hash(pred0[ins], det, rad)
        if len(pi) < 40:
            return None, sc, 0, 99.0
        J = np.zeros((2 * len(pi), 4))
        for k in range(3):
            v = np.zeros(3)
            v[k] = 1e-6
            p2F, _ = project_raw(Cc, M @ expm_rot(v))
            p2 = np.stack([CF.CX + sc * (p2F[:, 0] - CF.CX),
                           CF.CY + sc * (p2F[:, 1] - CF.CY)], 1)
            d = (p2 - pred0)[ins][pi]
            J[0::2, k] = d[:, 0] / 1e-6
            J[1::2, k] = d[:, 1] / 1e-6
        J[0::2, 3] = (predF[:, 0] - CF.CX)[ins][pi]
        J[1::2, 3] = (predF[:, 1] - CF.CY)[ins][pi]
        r = np.empty(2 * len(pi))
        r[0::2] = det[di, 0] - pred0[ins][pi, 0]
        r[1::2] = det[di, 1] - pred0[ins][pi, 1]
        sol, *_ = np.linalg.lstsq(J, r, rcond=None)
        acc = acc @ expm_rot(sol[:3])
        sc += sol[3]
    M = M0 @ acc
    predF, okF = project_raw(Cc, M)
    pred0 = np.stack([CF.CX + sc * (predF[:, 0] - CF.CX),
                      CF.CY + sc * (predF[:, 1] - CF.CY)], 1)
    ins = okF & (pred0[:, 0] > 0) & (pred0[:, 0] < CF.IMG_W) \
              & (pred0[:, 1] > 0) & (pred0[:, 1] < CF.IMG_H)
    pi, di, dd = match_hash(pred0[ins], det, 0.4)
    return acc, sc, len(pi), (float(np.median(dd)) if len(pi) else 99.0)


def calibrate_all(att0, frames, cat_dirs, G, gmax=9.0, topn=2500, verbose=True):
    """对多帧批量标定。att0: {key: M0}，返回 {key: M}。"""
    out = {}
    for k, M0 in att0.items():
        if k not in frames:
            continue
        st = frames[k]
        det = st[:, :2].astype(float)[np.argsort(st[:, 2])][:topn]
        acc, sc, nin, med = calib_frame_scale(np.array(M0), cat_dirs, G, det, gmax=gmax)
        if acc is None:
            print(f'  {k:>8} 失败')
            continue
        M = np.array(M0) @ acc
        out[k] = M
        if verbose:
            ang = np.degrees(np.arccos(np.clip((np.trace(acc) - 1) / 2, -1, 1)))
            print(f'  {k:>8} 内点 {nin:5d}  残差 {med:6.3f}px  转角 {ang:5.2f}°  '
                  f'尺度 {sc:.6f}')
    return out


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def solve_global_offset(M0, Cw, det, span=4.0, step=0.5, tol=6.0):
    """粗锚点通常差几度（手工对准 + 调平误差）。在 (dl, db, roll) 上搜索全局修正。

    返回 (G, 命中数, (dl, db, roll))，使 M = G @ M0。
    """
    best = (-1, None, (0.0, 0.0, 0.0))
    grid = np.arange(-span, span + 1e-9, step)
    for dl in grid:
        for db in grid:
            Gx = CF.GAL.T @ (rotz(np.radians(dl)) @ roty(np.radians(-db))) @ CF.GAL
            for roll in grid:
                M = Gx @ M0 @ rotz(np.radians(roll))
                pred, ok = project_raw(Cw, M)
                ins = ok & (pred[:, 0] > 0) & (pred[:, 0] < CF.IMG_W) \
                          & (pred[:, 1] > 0) & (pred[:, 1] < CF.IMG_H)
                if ins.sum() < 100:
                    continue
                pi, di, _ = match_hash(pred[ins], det, tol)
                if len(pi) > best[0]:
                    G = Gx @ M0 @ rotz(np.radians(roll)) @ np.linalg.inv(M0)
                    best = (len(pi), G, (float(dl), float(db), float(roll)))
    return best[1], best[0], best[2]


def solve_roll(M0, Cw, det, step=1.5, tol=6.0):
    """已知光轴、滚转未知：扫 0–360° 找最佳滚转。

    M0 是 det=+1 的相机->世界矩阵（光轴已知，滚转任意），
    M = M0 @ rotz(theta)。返回 (G, 命中数, theta_deg)，使 G @ M0 为最佳 M。
    粗扫用宽容差（step 造成的错位最多 step/2），再在峰值附近用 tol 精扫。
    """
    def score(theta, t):
        M = M0 @ rotz(np.radians(theta))
        pred, ok = project_raw(Cw, M)
        ins = ok & (pred[:, 0] > 0) & (pred[:, 0] < CF.IMG_W) \
                  & (pred[:, 1] > 0) & (pred[:, 1] < CF.IMG_H)
        if ins.sum() < 50:
            return 0
        pi, _di, _dd = match_hash(pred[ins], det, t)
        return len(pi)

    t_coarse = max(tol, 2.2 * step * 24.0)      # 1° 在角上约 24px
    thetas = np.arange(0.0, 360.0, step)
    sc = np.array([score(th, t_coarse) for th in thetas])
    best_i = int(np.argmax(sc))
    best = (int(sc[best_i]), float(thetas[best_i]))

    # 精扫：峰值附近 ±step，步长 step/15
    for th in np.arange(best[1] - step, best[1] + step + 1e-9, step / 15.0):
        n = score(th, tol)
        if n > best[0]:
            best = (n, float(th))
    th = best[1] % 360.0
    M = M0 @ rotz(np.radians(th))
    G = M @ np.linalg.inv(M0)
    return G, best[0], th
