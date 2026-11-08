"""假设-生成 + FFT 互相关的多尺度粗搜 x。

物理：给定候选 x，把星表前向投影 s_i(x)=(p_i-x)/|p_i-x|；
正确 x 下中距星重新聚拢，FFT 峰恢复（A 组反证：正确位置峰高
约为背景 10 倍）。FFT 对姿态平移免疫，故只需扫 x。
连续峰高绕开了逐星内点计数 ~2 pc 的窄吸引盆。
"""
import math
import numpy as np

from .. import register                     # astro-nav
from . import params as PR


def render(x, M, P, G, logd, mlim=None):
    """星表 -> 候选 x 视角的像素坐标 (u,v)。"""
    mlim = PR.COARSE_MLIM if mlim is None else mlim
    diff = P-x
    L = np.sqrt((diff*diff).sum(1))
    m = G + 5.0*(np.log10(np.maximum(L, 1e-9))-logd)
    q = np.where(m < mlim)[0]
    s = diff[q]/L[q, None]
    cc = s @ M
    z = cc[:, 2]
    ok = z > 0.12
    cc = cc[ok]
    uu = PR.FOC*cc[:, 0]/cc[:, 2]+PR.CX
    vv = PR.FOC*cc[:, 1]/cc[:, 2]+PR.CY
    ins = (uu > 0) & (uu < PR.IMG_W) & (vv > 0) & (vv < PR.IMG_H)
    return uu[ins], vv[ins]


def obs_map(det, bs):
    return register.splat(det[:, 0], det[:, 1],
                          PR.IMG_W//bs, PR.IMG_H//bs, bs)


def score(x, M, bs, obs, P, G, logd):
    u, v = render(x, M, P, G, logd)
    img = register.splat(u, v, PR.IMG_W//bs, PR.IMG_H//bs, bs)
    return register.corr_peak(img, obs)[0]


def multi_scale_search(det, M, P, G, logd, x0=None,
                       levels=None, n_keep=None, verbose=True):
    """从网格（或给定起点）出发逐级细化。返回 (x_coarse, diag)。

    x0=None 时做全空间 L0 网格；否则从 x0 附近出发（L0 跳过）。
    """
    levels = list(PR.COARSE_LEVELS if levels is None else levels)
    n_keep = PR.COARSE_N_KEEP if n_keep is None else n_keep
    obs = {bs: obs_map(det, bs) for bs, _, _ in levels}
    diag = dict(levels=[])
    if x0 is None:
        t0 = __import__('time').time()
        g = np.arange(-PR.RMAX_PC, PR.RMAX_PC+1e-9, PR.COARSE_GRID_PC)
        cand = []
        bs, _, _ = levels[0]
        for xv in g:
            for yv in g:
                for zv in g:
                    x = np.array([xv, yv, zv])
                    if np.linalg.norm(x) > PR.RMAX_PC:
                        continue
                    cand.append((score(x, M, bs, obs[bs], P, G, logd),
                                 x.copy()))
        cand.sort(key=lambda t: -t[0])
        vals = np.array([c[0] for c in cand])
        diag['levels'].append(dict(bin=bs, n=len(cand),
                                   med=float(np.median(vals)),
                                   max=float(vals.max())))
        cands = [x.copy() for _, x in cand[:n_keep]]
        if verbose:
            print(f'  coarse L0 bin={bs}: {len(cand)} 点 '
                  f'{__import__("time").time()-t0:.0f}s '
                  f'中位 {np.median(vals):.0f} 最大 {vals.max():.0f}',
                  flush=True)
    else:
        cands = [np.array(x0, float)]
    # 细化从 levels[1] 开始：levels[0] 只是 L0 网格的 bin，
    # 若重复使用会把低分辨率候选直接带进后续级（实测粗搜失败）
    best = None
    for li, (bs, step, ns) in enumerate(levels[1:], 1):
        out = []
        gg = np.arange(-ns, ns+1e-9, 1)*step
        for c in cands:
            for dx in gg:
                for dy in gg:
                    for dz in gg:
                        x = c+np.array([dx, dy, dz])
                        if np.linalg.norm(x) > PR.RMAX_PC:
                            continue
                        out.append((score(x, M, bs, obs[bs], P, G, logd),
                                    x.copy()))
        out.sort(key=lambda t: -t[0])
        cands = [x.copy() for _, x in out[:max(2, n_keep//2)]]
        best = out[0]
        diag['levels'].append(dict(bin=bs, peak=float(out[0][0]),
                                   x=out[0][1].tolist()))
        if verbose:
            print(f'  coarse L{li} bin={bs:2d} step={step:.1f}pc: '
                  f'峰 {out[0][0]:9.0f} 次优 {out[1][0]:9.0f} '
                  f'x={out[0][1].round(2)} '
                  f'|x|={np.linalg.norm(out[0][1])*PR.PC_LY:.2f}ly',
                  flush=True)
    # 峰/背景 比（判据：正确 x 应 ~10 倍，A 组反证标准）
    bsl = levels[-1][0]
    bg = score(np.zeros(3), M, bsl, obs[bsl], P, G, logd)
    diag['peak_over_zero'] = float(best[0]/max(bg, 1))
    if verbose:
        print(f'  coarse: 粗解峰/x=0背景 = {diag["peak_over_zero"]:.2f}',
              flush=True)
    return best[1], diag
