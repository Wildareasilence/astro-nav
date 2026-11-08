"""Wahba 定向：标称银道网格 -> 全局旋转 Q。

输入是用户给的拍摄网格（每圈纬度、起始银经、步进、银极归属）。
注意两点（都是实测教训）：
  1 dl（银经零点）与「绕银极的全局旋转」严格简并——扫描 dl 的
     拟合残差一字不变。绝对银经零点必须由外部锚点（"首帧大致
     对准银心，l≈0"）给定，绝不扫描。
  2 拟合只用各帧光轴方向；B（纬圈纬度）与 sgn（步进方向）可扫。
"""
import math
import numpy as np

from . import params as PR


def gal_unit(l_deg, b_deg):
    """银道 (l,b) -> ICRS 单位方向。"""
    l, b = math.radians(l_deg), math.radians(b_deg)
    lb = np.array([math.cos(b)*math.cos(l), math.cos(b)*math.sin(l),
                   math.sin(b)])
    return PR.AN.GAL.T @ lb


def gal_lb(w):
    lb = PR.AN.GAL @ np.asarray(w, float)
    l = math.degrees(math.atan2(lb[1], lb[0])) % 360
    b = math.degrees(math.asin(float(np.clip(lb[2], -1, 1))))
    return l, b


class GridSpec:
    """拍摄网格描述。

    bands : [(b_deg, l0_deg, step_deg, [frame_key, ...]), ...]
    poles : {'north': [keys], 'south': [keys]}
    """

    def __init__(self, bands, poles):
        self.bands = bands
        self.poles = poles

    def nominal(self, dl=0.0):
        """帧 -> 标称 ICRS 光轴方向。dl 为外部银经零点锚。"""
        out = {}
        for b, l0, step, keys in self.bands:
            for i, k in enumerate(keys):
                out[k] = gal_unit(l0+dl+step*i, b)
        for k in self.poles.get('north', []):
            out[k] = PR.AN.GAL[2].copy()
        for k in self.poles.get('south', []):
            out[k] = -PR.AN.GAL[2].copy()
        return out


def kabsch(C, Uv):
    """行向量: C @ Q = Uv。"""
    Bm = C.T @ Uv
    U, _s, Vt = np.linalg.svd(Bm)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1., 1., d]) @ Vt


def solve_wahba(R_rel, grid, dl=None, b_grid=None, verbose=True):
    """拼接系 -> 全局旋转 Q（w_icrs = w_mosaic @ Q）。

    R_rel : {key: 相对旋转}，来自 mosaic.build_mosaic
    dl    : 外部银经零点锚（度）。缺省用 params.WAHBA_DL。
    返回 (Q, M_frames, diag)。M_k 满足 s_icrs @ M_k = cam。
    """
    if dl is None:
        dl = PR.WAHBA_DL
    if b_grid is None:
        b_grid = PR.WAHBA_B_GRID
    cams_axis = {k: np.array([0., 0., 1.]) @ R.T for k, R in R_rel.items()}
    best = (-1e9, None, None)
    for Bv in b_grid:
        for sgn in (1, -1):
            # Bv 替换非零纬度圈的幅值，但保持原符号（下圈 b=-Bv）；
            # 不做这一步，下圈标称会在 +Bv 处，残差 ~2B 度（实测 80°）
            g = GridSpec([(math.copysign(Bv, b) if abs(b) > 1e-9 else b,
                           l0, sgn*abs(step), keys)
                          for (b, l0, step, keys) in grid.bands],
                         grid.poles)
            nom = g.nominal(dl)
            ks = [k for k in nom if k in cams_axis]
            C = np.array([cams_axis[k] for k in ks])
            Uv = np.array([nom[k] for k in ks])
            Q = kabsch(C, Uv)
            res = np.degrees(np.arccos(np.clip(
                np.einsum('ij,ij->i', C @ Q, Uv), -1, 1)))
            sc = -float(np.median(res))
            if sc > best[0]:
                best = (sc, Q, dict(B=Bv, sgn_l=sgn, dl=dl,
                                    med_deg=float(np.median(res)),
                                    p90_deg=float(np.percentile(res, 90)),
                                    n_frames=len(ks)))
    _, Q, diag = best
    M = {k: Q.T @ R for k, R in R_rel.items()}
    if verbose:
        print(f'  wahba: B={diag["B"]} sgn={diag["sgn_l"]} dl={dl} '
              f'残差 中位 {diag["med_deg"]:.3f}° p90 {diag["p90_deg"]:.3f}° '
              f'({diag["n_frames"]} 帧)', flush=True)
    return Q, M, diag


def check_anchor(M, anchor_key, l_expect=0.0):
    """诊断：锚点帧对齐后的实际 (l,b)，验证外部锚是否合理。"""
    l, b = gal_lb(M[anchor_key][:, 2])
    return dict(key=anchor_key, l=l, b=b,
                offset_deg=math.hypot((l-l_expect+180) % 360-180, b))
