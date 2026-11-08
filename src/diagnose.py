"""三个诊断：判断系统差属于哪一类。

1) stratify     按星表距离上限分层，看 |x| 是否随上限增长
2) roll_pair    同一天区不同滚转，看残差图案固定在探测器还是天球
3) epoch_scan   改星表历元，看 |x| 是否在某个历元取极小
"""
import numpy as np

import config as CF
from src import sky, solver, attitude


def stratify(att, frames, cat, cutoffs=(40, 70, 120, 200), tag=''):
    """若答案随距离上限变化 -> 有"恒定角度 × 距离"的系统差；若不变 -> 真视差。"""
    print(f'[{tag}] 距离分层：')
    print(f'{"上限(pc)":>9} {"匹配星":>7} {"|x| (AU)":>10}')
    for dc in cutoffs:
        # 只保留距离上限内的星
        idx = np.where(np.isfinite(cat['pos']).all(1))[0]
        dist = np.linalg.norm(cat['pos'], axis=1)
        keep = np.zeros(len(cat['G']), bool)
        keep[idx[dist[idx] < dc * CF.AU_PER_PC]] = True
        sub = sky.subset(cat, np.where(keep)[0])
        rows, Pw, _ = solver.build_rows(att, frames, sub)
        n = sum(len(r[0]) for r in rows)
        if n < 500:
            print(f'{dc:9d} {n:7d}       --')
            continue
        p, cov, c2 = solver.fit(rows, Pw, 'x')
        print(f'{dc:9d} {n:7d} {np.linalg.norm(p[:3]):10.1f}')
    print('  → 单调增长说明存在 ∝d 的系统差（恒定角度偶极）。')


def residual_map(M, Cw, det, nbx=8, nby=5, rad=0.6):
    """在探测器坐标下返回每块的平��残差向量。"""
    pred, ok = attitude.project_raw(Cw, M)
    ins = ok & (pred[:, 0] > 0) & (pred[:, 0] < CF.IMG_W) \
              & (pred[:, 1] > 0) & (pred[:, 1] < CF.IMG_H)
    pi, di, _ = attitude.match_hash(pred[ins], det, rad)
    p = pred[ins][pi]
    o = det[di]
    rx, ry = o[:, 0] - p[:, 0], o[:, 1] - p[:, 1]
    rx -= rx.mean()
    ry -= ry.mean()
    bx = np.clip((p[:, 0] / (CF.IMG_W / nbx)).astype(int), 0, nbx - 1)
    by = np.clip((p[:, 1] / (CF.IMG_H / nby)).astype(int), 0, nby - 1)
    mx = np.zeros((nby, nbx)); my = np.zeros((nby, nbx)); cn = np.zeros((nby, nbx))
    np.add.at(mx, (by, bx), rx)
    np.add.at(my, (by, bx), ry)
    np.add.at(cn, (by, bx), 1)
    return mx / np.maximum(cn, 1), my / np.maximum(cn, 1), cn, len(pi)


def roll_pair(att, frames, cat, a, b, gmax=9.0):
    """同一片天、不同滚转的两帧。残差向量若在探测器坐标下一致 -> 探测器问题。"""
    n0, _, _ = sky.project(np.zeros(3), cat, None, CF.GAL_EPOCH)
    Cw = n0[cat['G'] < gmax]
    maps = {}
    for k in (a, b):
        det = frames[k][:, :2].astype(float)[np.argsort(frames[k][:, 2])][:3000]
        maps[k] = residual_map(np.array(att[k]), Cw, det)
    va = np.stack([maps[a][0], maps[a][1]], -1)
    vb = np.stack([maps[b][0], maps[b][1]], -1)
    good = (maps[a][2] > 20) & (maps[b][2] > 30)
    if good.sum() < 8:
        print(f'{a}/{b}: 有效块太少')
        return None
    va_, vb_ = va[good], vb[good]
    cosang = (va_ * vb_).sum(1) / np.maximum(
        np.linalg.norm(va_, axis=1) * np.linalg.norm(vb_, axis=1), 1e-12)
    med = np.degrees(np.arccos(np.clip(np.median(cosang), -1, 1)))
    print(f'{a}/{b}: 有效块 {int(good.sum())}  残差向量夹角中位 {med:.0f}°')
    print(f'   探测器固定 -> 约 0°；天球固定 -> 约两帧滚转差')
    return med


def epoch_scan(att, frames, cat, epochs=(1950, 2000, 2026, 2100, 2200, 2300)):
    """改变星表传播历元，看 |x| 是否在某个历元降到 0。"""
    print(f'{"历元":>7} {"匹配星":>7} {"|x| (AU)":>10}')
    for ep in epochs:
        old = CF.GAL_EPOCH
        CF.GAL_EPOCH = float(ep)
        try:
            rows, Pw, _ = solver.build_rows(att, frames, cat)
            n = sum(len(r[0]) for r in rows)
            p, cov, c2 = solver.fit(rows, Pw, 'x')
            print(f'{ep:7d} {n:7d} {np.linalg.norm(p[:3]):10.1f}')
        finally:
            CF.GAL_EPOCH = old
