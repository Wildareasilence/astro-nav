"""帧间拼接：由配对旋转递推每帧相对姿态，并给出银道锚点。

约定（务必保持）：
  · 配对给出的 R 满足   cam_a @ R = cam_b
  · 递推必须是         R_b = R_a @ R
  · 相机系是左手系（x 右、y 下、z 前），需全局翻 x：
        相机 -> 世界   M_k = A @ FLIP @ R[k]
    FLIP 必须在 R 的左边。写成 A @ R @ FLIP 只会把经度镜像、纬度不变，
    环结构看起来仍然正确，极难发现。
"""
import numpy as np

import config as CF

FLIP = np.diag([-1.0, 1.0, 1.0])


def gal_unit(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    lb = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    return CF.GAL.T @ lb


def anchor_R(l0=0.0, b0=0.0, flip_east=False, up_pole=True):
    """粗锚点：光轴指向 (l0,b0)，画面 x 轴沿银道面，画面上方朝银北极。

    返回的是*非正交右手*矩阵（det = -1），配合 FLIP 使用。
    """
    z = gal_unit(l0, b0)
    east = gal_unit(l0 + 90.0, 0.0)
    east = east - (east @ z) * z
    east /= np.linalg.norm(east)
    if flip_east:
        east = -east
    npp = CF.GAL[2] - (CF.GAL[2] @ z) * z
    npp /= np.linalg.norm(npp)
    y = -npp if up_pole else npp
    return np.stack([east, y, z], axis=1)


def anchor_radec(ra_deg, dec_deg):
    """已知某一帧光轴的赤经赤纬时的粗锚点，滚转待定（det = -1，配 FLIP 用）。

    与 anchor_R 同一手性约定：某帧世界矩阵 M = anchor_radec(...) @ FLIP @ R[k]。
    滚转留给 attitude.solve_roll 扫（0–360°），因为手工摆平可能有任意角度。
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    z = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    npp = CF.GAL[2] - (CF.GAL[2] @ z) * z
    npp /= np.linalg.norm(npp)
    y = -npp
    east = np.cross(z, y)
    east /= np.linalg.norm(east)
    return np.stack([east, y, z], axis=1)


def frame_M(l_deg, b_deg):
    """一个标准朝向的相机->世界旋转（det = +1）：光轴在 (l,b)，银道水平、北朝上。"""
    z = gal_unit(l_deg, b_deg)
    east = gal_unit(l_deg + 90.0, 0.0)
    east = east - (east @ z) * z
    east /= np.linalg.norm(east)
    north = np.cross(z, east)
    return np.stack([-east, -north, z], axis=1)


def propagate(pairs, keys, start=None):
    """pairs 每项含 a,b,R。返回 (R, seen, order)，R[k] 为相机k->相机seed。"""
    adj = {}
    for p in pairs:
        Rab = np.array(p['R'])
        adj.setdefault(p['a'], []).append((p['b'], Rab))
        adj.setdefault(p['b'], []).append((p['a'], Rab.T))   # 反向必须转置
    if start is None:
        start = keys[0] if keys[0] in adj else next(iter(adj))
    R = {start: np.eye(3)}
    seen = {start}
    st = [start]
    order = [start]
    while st:
        a = st.pop()
        for b, Rab in adj.get(a, []):
            if b in seen:
                continue
            R[b] = R[a] @ Rab            # ← 唯一正确写法
            seen.add(b)
            st.append(b)
            order.append(b)
    return R, seen, order


def attitude_matrix(Rk, A=None):
    """由相对姿态得到相机->世界的完整矩阵。"""
    if A is None:
        A = anchor_R(0.0, 0.0, False, True)
    return A @ FLIP @ Rk


def closure_check(pairs, tol_print=True):
    """三帧环路闭合检验：Rab @ Rbc 应等于 Rac。"""
    d = {(p['a'], p['b']): np.array(p['R']) for p in pairs}
    def Rab(a, b):
        if (a, b) in d:
            return d[(a, b)]
        if (b, a) in d:
            return d[(b, a)].T
        return None
    errs = []
    for a, b in list(d):
        for c in set([x for (x, _y) in d if x == a or x == b]):
            if c in (a, b):
                continue
            R1, R2, R3 = Rab(a, b), Rab(a, c), Rab(b, c)
            if R1 is None or R2 is None or R3 is None:
                continue
            ang = np.degrees(np.arccos(
                np.clip((np.trace((R1 @ R3).T @ R2) - 1) / 2, -1, 1)))
            errs.append(ang)
    if not errs:
        return None
    if tol_print:
        print(f'闭合检验 {len(errs)} 组：中位 {np.median(errs):.4f}°  '
              f'p90 {np.percentile(errs, 90):.4f}°  最大 {max(errs):.3f}°')
    return np.array(errs)


def frame_M_radec(ra_deg, dec_deg):
    """已知光轴指向 (ra,dec) 的标准朝向：相机->世界，银河北朝上（det = +1）。

    与 frame_M 同一约定，M_k = M_seed @ R[k]，不需要 FLIP。
    滚转用 M = frame_M_radec(...) @ rotz(theta)。
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    z = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    npp = CF.GAL[2] - (CF.GAL[2] @ z) * z
    npp /= np.linalg.norm(npp)
    down = -npp
    right = np.cross(down, z)
    right /= np.linalg.norm(right)
    return np.stack([right, down, z], axis=1)
