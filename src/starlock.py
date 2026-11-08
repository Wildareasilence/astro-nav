"""完全盲解：不认识银河、不知道任何方向，只用「相机 + 三维星表」定姿态。

思路（经典星敏感器算法，但用全天星表做）
  1. 取一帧最亮的 ~60 颗星（要求彼此 >0.8°，避免双星/混合像元）。
  2. 假设其中某一颗是星表里的某颗已知亮星，看「从它到其余 39 颗的角距集合」
     能否在星表里找到同样的集合。这一步是旋转不变的，只用到角距，不用亮度顺序
     以外的任何信息。
  3. 一旦锁住一颗（锚星），再用第二颗星定绕锚星的方位角 → 得到一个完整旋转。
  4. 拿这个旋转去投影全部星表亮星，数落在实拍星点上的个数。对的那一个会有几百，
     错的只有几个。
  5. 交 attitude.calibrate_frame 精修。

为什么可行：这是「丢失在太空中」问题的标准解法。亮度顺序经实测与星表一致
（G 2–5 的星正好是检出最亮的前 200 颗），所以两边的「最亮 60 颗」是同一批天体。
"""
import numpy as np

import config as CF
from src import prep, attitude

ARCSEC = np.pi / (180 * 3600)


def unit_of_uv(uv):
    return prep.pix2cam(uv)


def ang_between(a, b):
    return np.arccos(np.clip(a @ b, -1.0, 1.0))


def pick_bright(dirs, n=60, min_sep_deg=0.8, cand=None):
    """按给定顺序取 n 颗，要求两两角距 > min_sep_deg。"""
    if cand is None:
        cand = range(len(dirs))
    out = []
    ms = np.radians(min_sep_deg)
    for i in cand:
        if all(ang_between(dirs[i], dirs[j]) > ms for j in out):
            out.append(i)
            if len(out) >= n:
                break
    return np.array(out, dtype=int)


def sep_list(u, i0, others):
    """从第 i0 颗到 others 的角距（弧度）。"""
    return np.arccos(np.clip(u[others] @ u[i0], -1.0, 1.0))


def count_coincide(s_ref, s_cat_sorted, tol):
    """s_ref 里每个值，在 s_cat_sorted 中 tol 内是否有一个对应。"""
    lo = np.searchsorted(s_cat_sorted, s_ref - tol, 'left')
    hi = np.searchsorted(s_cat_sorted, s_ref + tol, 'right')
    return int((hi > lo).sum())


def align_two(o1, o2, c1, c2):
    """构造把 (o1,o2) 转到 (c1,c2) 的旋转（最小旋转 + 绕 c1 的方位修正）。"""
    R0 = attitude_align(o1, c1)
    y = R0 @ o2
    # y 与 c2 都在 ⟂ c1 的平面内，投到该平面再求夹角
    def proj(x):
        z = x - (x @ c1) * c1
        n = np.linalg.norm(z)
        return z / n if n > 1e-12 else None
    py, pc = proj(y), proj(c2)
    if py is None or pc is None:
        return R0
    ang = np.arctan2(np.cross(py, pc) @ c1, py @ pc)
    return attitude_rot(c1, ang) @ R0


def attitude_align(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999999:
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def attitude_rot(n, ang):
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def kabsch(p, q):
    """求旋转 R 使 q ≈ p @ R.T（行矢量）。"""
    B = p.T @ q
    U, _S, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return (U @ D @ Vt).T


def polish(M, obs_dirs, cat_dirs, iters=5, tol_arcsec=900.0):
    """用匹配上的星对做 Kabsch，把粗解从角分压到角秒。"""
    for _ in range(iters):
        cc = cat_dirs @ M
        z = cc[:, 2]
        ok = z > 0.2
        u = CF.FOC * cc[:, 0] / np.maximum(z, 1e-9) + CF.CX
        v = CF.FOC * cc[:, 1] / np.maximum(z, 1e-9) + CF.CY
        ins = ok & (u > 0) & (u < CF.IMG_W) & (v > 0) & (v < CF.IMG_H)
        if ins.sum() < 20:
            break
        ou = obs_dirs[:, 0] / obs_dirs[:, 2] * CF.FOC + CF.CX
        ov = obs_dirs[:, 1] / obs_dirs[:, 2] * CF.FOC + CF.CY
        ci_ins = np.where(ins)[0]
        pu, pv = u[ins], v[ins]
        d = np.sqrt((pu[:, None] - ou[None, :]) ** 2 + (pv[:, None] - ov[None, :]) ** 2)
        j = d.argmin(1)
        dmin = d[np.arange(len(pu)), j]
        k = dmin < tol_arcsec / CF.ARCSEC_PX
        if k.sum() < 12:
            break
        # 一颗观测星只保留最近的一颗星表星
        best = {}
        for c, o, dd in zip(ci_ins[k], j[k], dmin[k]):
            if o not in best or dd < best[o][0]:
                best[o] = (dd, c)
        oi = np.array(list(best.keys()))
        ci = np.array([best[o][1] for o in oi])
        if len(ci) < 12:
            break
        M_new = kabsch(obs_dirs[oi], cat_dirs[ci])
        done = np.allclose(M_new, M, atol=1e-10)
        M = M_new
        if done:
            break
        tol_arcsec = max(150.0, tol_arcsec * 0.45)
    return M


def verify(M, obs_dirs, cat_dirs, tol_arcsec=10.0):
    """把 cat 投影进相机，数有多少落在 obs 上。返回 (命中数, 中位残差弧秒)。"""
    cc = cat_dirs @ M
    z = cc[:, 2]
    ok = z > 0.2
    u = CF.FOC * cc[:, 0] / np.maximum(z, 1e-9) + CF.CX
    v = CF.FOC * cc[:, 1] / np.maximum(z, 1e-9) + CF.CY
    ins = ok & (u > 0) & (u < CF.IMG_W) & (v > 0) & (v < CF.IMG_H)
    if ins.sum() < 20:
        return 0, 999.0
    ou = obs_dirs[:, 0] / obs_dirs[:, 2] * CF.FOC + CF.CX
    ov = obs_dirs[:, 1] / obs_dirs[:, 2] * CF.FOC + CF.CY
    pu, pv = u[ins], v[ins]
    d = np.sqrt((pu[:, None] - ou[None, :]) ** 2 + (pv[:, None] - ov[None, :]) ** 2)
    dmin = d.min(1)
    tol_px = tol_arcsec / CF.ARCSEC_PX
    hit = int((dmin < tol_px).sum())
    return hit, float(np.median(dmin[dmin < tol_px]) * CF.ARCSEC_PX) if hit else 999.0


def blind_attitude(obs_uv, cat_dirs, cat_G, cat_pos=None, gmax_pair=7.0, n_pair_stars=400,
                   n_obs_try=6, tol_pair_arcsec=120.0, verify_tol_arcsec=500.0,
                   dmin_pc=0.0, verbose=True):
    """完全盲解：只用「最亮几颗星之间的大角距」做种子，再用全场计数验证。

    obs_uv: (N,2) 该帧星点，按亮度降序。
    返回 (M, 诊断)，M 把该帧相机系转到天球（ICRS）。
    """
    obs = unit_of_uv(obs_uv)
    nb = min(80, len(obs))
    cand = pick_bright(obs, n=nb, min_sep_deg=0.8)
    bright = obs[cand]

    keep = np.ones(len(cat_G), bool)
    if dmin_pc > 0:
        keep = np.linalg.norm(cat_pos, axis=1) > dmin_pc * CF.AU_PER_PC
    sel = (cat_G < gmax_pair) & keep
    Cb = cat_dirs[sel]
    if len(Cb) > n_pair_stars:
        Cb = Cb[np.argsort(cat_G[sel])[:n_pair_stars]]
    Dcat = np.arccos(np.clip(Cb @ Cb.T, -1, 1))
    if verbose:
        print(f'  观测亮星 {len(bright)} 颗；配对用星表 {len(Cb)} 颗（G<{gmax_pair:.1f}）')

    # 验证用：全部星表（到 G<9）+ 观测前 400
    selv = (cat_G < 9.0) & keep
    Cv = cat_dirs[selv]
    ov = bright

    tol = tol_pair_arcsec * ARCSEC
    trials = []
    npair = min(n_obs_try, len(bright) - 1)
    for ia in range(npair):
        for ib in range(ia + 1, min(ia + 5, len(bright))):
            o1, o2 = bright[ia], bright[ib]
            s12 = np.arccos(np.clip(o2 @ o1, -1, 1))
            ii, jj = np.where(np.abs(Dcat - s12) < tol)
            m = ii < jj
            for c1, c2 in zip(ii[m], jj[m]):
                for (a, b, p, q) in ((o1, o2, c1, c2), (o2, o1, c1, c2)):
                    M = align_two(a, b, Cb[p], Cb[q])
                    hit, res = verify(M, ov, Cv, verify_tol_arcsec)
                    trials.append((hit, res, M, ia, ib, int(p), int(q)))
    if not trials:
        return None, {}
    trials.sort(key=lambda t: (-t[0], t[1]))
    hit, res, M, ia, ib, p, q = trials[0]
    M = polish(M, ov, Cv)
    hit2, res2 = verify(M, ov, Cv, verify_tol_arcsec)
    if res2 < res or hit2 > hit:
        hit, res = hit2, res2
    if verbose:
        print(f'  {len(trials)} 个种子，最佳命中 {hit}/{len(ov)}  中位 {res:.0f}″ '
              f'(随机 ≈ {len(ov) * len(Cv) * np.pi * (verify_tol_arcsec / CF.ARCSEC_PX) ** 2 / (CF.IMG_W * CF.IMG_H):.1f})')
    return M, dict(hit=hit, res_arcsec=res, n_trials=len(trials),
                   pair=(ia, ib, p, q))


# 位移越大，能用来匹配的星必须越远（位移让距离 L 的星挪 |x|/L）。
# 每一档给出 (dmin_pc, 容差角秒)。容差 ≈ |x|/dmin 的量级。
PRESETS = [(0.0, 120.0), (200.0, 300.0), (300.0, 700.0),
           (500.0, 1500.0), (800.0, 3000.0)]


def blind_auto(obs_uv, cat_dirs, cat_G, cat_pos, presets=None,
               score_tol_arcsec=400.0, verbose=True):
    """**实验性**：逐档试，用「紧容差下的真实命中数」挑最好的。

    注意：只在 |x| ≲ 1 ly 验证过（0 ly 时正确档给出 154 命中，其它档 18–37）。
    |x| ≥ 3 ly 时各档的命中数都掉到同一量级，挑选会失效——这是图案本身的畸变，
    不是打分函数的问题。超出 1 ly 请改用联合搜索（见 docs/pipeline.md §12）。

    判据不能只看残差：在错的档上，算法会在容差内硬凑出一个残差 ≈ 容差的解。
    所以每档解完都用**不筛距离的全星表**、**紧容差**再数一次命中——
    真解会有一批星落进 400″，假解几乎没有。
    """
    obs = unit_of_uv(obs_uv)
    nb = min(80, len(obs))
    bright = obs[pick_bright(obs, n=nb, min_sep_deg=0.8)]
    out = []
    for dmin, tol in (presets or PRESETS):
        try:
            M, d = blind_attitude(obs_uv, cat_dirs, cat_G, cat_pos=cat_pos,
                                  dmin_pc=dmin, tol_pair_arcsec=tol,
                                  verify_tol_arcsec=tol * 1.5,
                                  n_pair_stars=300, verbose=False)
        except Exception:
            continue
        if M is None:
            continue
        hit, res = verify(M, bright, cat_dirs, score_tol_arcsec)
        out.append((hit, res, dmin, tol, M, d))
        if verbose:
            print(f'    dmin={dmin:5.0f}pc 容差={tol:5.0f}″ -> 紧容差命中 {hit:4d}  '
                  f'残差 {res:6.0f}″')
    if not out:
        return None, {}
    out.sort(key=lambda t: -t[0])
    hit, res, dmin, tol, M, d = out[0]
    if verbose:
        print(f'  采用 dmin={dmin:.0f}pc 容差={tol:.0f}″（紧容差命中 {hit}）')
    return M, dict(d, dmin_pc=dmin, tol_arcsec=tol, tight_hit=hit, tight_res=res)
