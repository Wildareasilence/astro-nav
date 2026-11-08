"""位置与光行差联合解算。

模型：
    w_pred = normalize( unit(p_i - x) + (I - w wᵀ)·v/c )

· 位置项 x  产生 ∝ 1/d 的角度偶极（视差）
· 光行差 v 产生与距离无关的角度偶极
两者形状相同，只能靠星表距离跨度分开。务必三个模型都跑一遍比较 χ²。
"""
import numpy as np

import config as CF
from src import sky, attitude

SIG_PX = 0.35          # 单星质心噪声（像素），用实际残差重标定


def build_rows(att, frames, cat, gmax=10.0, match_rad=0.6):
    """把每帧的匹配星整理成 (星表索引, 观测方向)。"""
    pos = cat['pos']
    n0, _, _ = sky.project(np.zeros(3), cat, None, CF.GAL_EPOCH)
    G = cat['G']
    sel = np.isfinite(pos).all(1) & (G < gmax)
    idx = np.where(sel)[0]
    rows = []
    for k, M in att.items():
        if k not in frames:
            continue
        M = np.array(M)
        uv = frames[k][:, :2].astype(float)
        pred, ok = attitude.project_raw(n0[idx], M)
        ins = ok & (pred[:, 0] > 0) & (pred[:, 0] < CF.IMG_W) \
                  & (pred[:, 1] > 0) & (pred[:, 1] < CF.IMG_H)
        gi = np.where(ins)[0]
        pi, di, _ = attitude.match_hash(pred[ins], uv, match_rad)
        if len(pi) < 100:
            continue
        o = uv[di]
        cam = np.stack([(o[:, 0] - CF.CX) / CF.FOC,
                        (o[:, 1] - CF.CY) / CF.FOC,
                        np.ones(len(o))], 1)
        cam /= np.linalg.norm(cam, axis=1, keepdims=True)
        rows.append((gi[pi], cam @ M.T))
    return rows, pos[idx], G[idx]


def fit(rows, Pw, mode='both', sig_px=SIG_PX, iters=15):
    """mode: 'x' 只解位置 / 'v' 只解光行差 / 'both' 两者。

    返回 (p, cov, chi2_per_N)，p = [x(AU), v(km/s)]。
    """
    sig = sig_px / CF.FOC
    idxmap = {'x': [0, 1, 2], 'v': [3, 4, 5], 'both': [0, 1, 2, 3, 4, 5]}
    ii = idxmap[mode]
    p = np.zeros(6)
    A = np.zeros((6, 6))
    chi = n = 0.0
    for _ in range(iters):
        A = np.zeros((6, 6))
        b = np.zeros(6)
        chi = n = 0.0
        for gi, w_obs in rows:
            x, v = p[:3], p[3:]
            d = Pw[gi] - x
            dn = np.linalg.norm(d, axis=1)
            w0 = d / dn[:, None]
            Pm = np.eye(3)[None, :, :] - w0[:, :, None] * w0[:, None, :]
            w = w0 + (Pm @ v[None, :, None])[:, :, 0] / CF.C_KMS
            w /= np.linalg.norm(w, axis=1)[:, None]
            r = w_obs - w
            r = r - (r * w).sum(1)[:, None] * w
            J = np.concatenate([-Pm / dn[:, None, None], Pm / CF.C_KMS], 2)
            A += np.einsum('nij,nik->jk', J, J) / sig ** 2
            b += np.einsum('nij,ni->j', J, r) / sig ** 2
            chi += (r ** 2).sum() / sig ** 2
            n += len(gi)
        sub = A[np.ix_(ii, ii)]
        p[ii] += np.linalg.solve(sub, b[ii])
    cov = np.zeros((6, 6))
    cov[np.ix_(ii, ii)] = np.linalg.inv(sub)
    return p, cov, chi / max(n, 1)


def report(p, cov, chi2n, n, tag=''):
    x, v = p[:3], p[3:]
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    print(f'[{tag}] χ²/N = {chi2n:.4f}   N = {n}')
    print(f'     位置 x = ({x[0]:+.1f} ± {sd[0]:.1f}, {x[1]:+.1f} ± {sd[1]:.1f}, '
          f'{x[2]:+.1f} ± {sd[2]:.1f}) AU   |x| = {np.linalg.norm(x):.1f} AU')
    if np.linalg.norm(v) > 0:
        g = CF.GAL @ (v / np.linalg.norm(v))
        l = np.degrees(np.arctan2(g[1], g[0])) % 360
        b = np.degrees(np.arcsin(np.clip(g[2], -1, 1)))
        print(f'     光行差 v = ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}) km/s   '
              f'|v| = {np.linalg.norm(v):.2f} km/s   l={l:.1f}° b={b:+.1f}°')
    return x, v
