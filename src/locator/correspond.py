"""对应建立与姿态精修：当前解 x -> 每帧姿态 -> 3px 对应。

关键原则（本项目实测的最大误差源）：
  对应必须随「最新解」重建。若用陈旧粗解投影并固定对应集合，
  会被粗解的自我实现偏差拉住（实测单帧 0.025 -> 0.006 ly）。
"""
from collections import defaultdict

import numpy as np

from .. import prep, attitude                # astro-nav
from . import params as PR


def skeleton(P, G, logd, x):
    """用当前解生成骨架星方向（只用于 calib 姿态，不用于建对应）。"""
    diff = P-x
    L = np.sqrt((diff*diff).sum(1))
    m = G+5*(np.log10(np.maximum(L, 1e-9))-logd)
    q = np.where(m < PR.SKELETON_GMAX)[0]
    q = q[np.argsort(G[q])[:PR.SKELETON_NMAX]]
    return diff[q]/L[q, None], G[q]


def project_frame(x, M, P, G, logd, mlim=None, nmax=20000):
    """星表 -> (像素, 星表索引, 视星等)。"""
    mlim = PR.MODEL_MLIM if mlim is None else mlim
    diff = P-x
    L = np.sqrt((diff*diff).sum(1))
    m = G+5*(np.log10(np.maximum(L, 1e-9))-logd)
    q = np.where(m < mlim)[0]
    s = diff[q]/L[q, None]
    cc = s @ M
    ok = cc[:, 2] > 0.12
    cc, qq, mm = cc[ok], q[ok], m[q][ok]
    uu = PR.FOC*cc[:, 0]/cc[:, 2]+PR.CX
    vv = PR.FOC*cc[:, 1]/cc[:, 2]+PR.CY
    ins = (uu > 3) & (uu < PR.IMG_W-3) & (vv > 3) & (vv < PR.IMG_H-3)
    uu, vv, qq, mm = uu[ins], vv[ins], qq[ins], mm[ins]
    if len(uu) > nmax:
        t = np.argsort(mm)[:nmax]
        uu, vv, qq = uu[t], vv[t], qq[t]
    return np.stack([uu, vv], 1), qq


def correspond_frame(x, M, det, P, G, logd, rad=None, mlim=None):
    """单帧 3px 对应。返回 (观测行号, 星表索引, 模型数)。"""
    rad = PR.CORRESPOND_RAD_PX if rad is None else rad
    pred, qi = project_frame(x, M, P, G, logd, mlim=mlim)
    nb = 8
    bk = defaultdict(list)
    for j, (u, v) in enumerate(det):
        bk[(int(u//nb), int(v//nb))].append(j)
    oj, cidx = [], []
    for i, (u, v) in enumerate(pred):
        cu, cv = int(u//nb), int(v//nb)
        best, bj = rad*rad, -1
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                for j in bk.get((cu+ax, cv+ay), ()):
                    d2 = (det[j, 0]-u)**2+(det[j, 1]-v)**2
                    if d2 < best:
                        best, bj = d2, j
        if bj >= 0:
            oj.append(bj); cidx.append(qi[i])
    return np.array(oj, np.int64), np.array(cidx, np.int64), len(pred)


def build_correspondences(frames, M_frames, x, P, G, D, logd,
                          verbose=False):
    """逐帧：calib 精修姿态 -> 3px 对应 -> 观测方向转 ICRS。

    返回 (W, ci, groups, per_frame)。W = 相机方向 @ M_k.T。
    """
    sk, sg = skeleton(P, G, logd, x)
    Wl, Cl, groups, ptr, per = [], [], [], 0, []
    for k in sorted(frames):
        fr = frames[k]
        det = fr[:, :2].astype(float)[
            np.argsort(fr[:, 2])[:PR.DETS_PER_FRAME]]
        acc, sc, nin, med = attitude.calib_frame_scale(
            M_frames[k], sk, sg, det[:4000], gmax=PR.CALIB_GMAX, iters=10)
        M = M_frames[k] if acc is None else M_frames[k] @ acc
        oj, cidx, _ = correspond_frame(x, M, det, P, G, logd)
        if len(oj) < 100:
            per.append(dict(key=k, n=0, calib_res=99.))
            continue
        cam = prep.pix2cam(det[oj])
        Wl.append(cam @ M.T)
        Cl.append(cidx)
        groups.append(np.arange(ptr, ptr+len(oj))); ptr += len(oj)
        per.append(dict(key=k, n=len(oj),
                        calib_res=(99. if acc is None else float(med)),
                        calib_nin=int(nin)))
        if verbose:
            print(f'    {k}: 对应 {len(oj)}  calib {per[-1]["calib_res"]:.2f}px',
                  flush=True)
    return np.concatenate(Wl), np.concatenate(Cl), groups, per
