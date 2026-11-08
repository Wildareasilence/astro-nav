"""步骤 3：拼接 + 绝对定向 + 逐帧精修。

    python scripts/03_attitude.py A     # 用银道锚点自动定全局朝向
    python scripts/03_attitude.py B     # 通过银道圈帧继承 A 组的绝对姿态

输出 results/att_<name>.json
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets as DS
import config as CF
from src import prep, mosaic, attitude, sky, register as RG, starlock as SL


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'A'
    d = DS.DATASETS[name]
    fr = prep.load_frames(d['out'])
    keys = sorted(fr)
    pairs = json.load(open(os.path.join(CF.RESULT_DIR, f'pairs_{name}.json')))

    LINK_GRP = ('link', 'old')      # 'old' 是旧版配对文件里对旧组帧的连线
    inner = [p for p in pairs if p.get('grp') not in LINK_GRP]
    R, seen, _ = mosaic.propagate(inner, keys)
    print(f'拼接：连通 {len(seen)}/{len(keys)}')
    mosaic.closure_check(inner)

    # 三种绝对定向方式：数据集给了某帧的赤经赤纬 -> 直接建 M 并扫滚转；
    # 有与 A 组的链接边 -> 继承 A 组姿态；否则用银心朝向 + 全局搜索。
    anc = d.get('anchor')
    if anc:
        att0 = None                       # 在下面 anchor 分支里直接算
    else:
        att0 = {k: mosaic.attitude_matrix(R[k]) for k in keys if k in R}

    cat = CF.load_catalog()
    n0, _, _ = sky.project(np.zeros(3), cat, None, CF.GAL_EPOCH)
    G = cat['G']

    link = [p for p in pairs if p.get('grp') in LINK_GRP]
    if anc:
        key0 = anc.get('frame')
        if key0 not in R:
            key0 = keys[0] if keys[0] in R else next(iter(R))
        det = fr[key0][:, :2].astype(float)[np.argsort(fr[key0][:, 2])][:3000]
        Cb = n0[G < 6.5]
        Cc = n0[G < 9.0]
        # 粗姿态：要么给了赤经赤纬，要么只给了「大致对着某个天体」
        if 'ra_deg' in anc:
            M0 = mosaic.frame_M_radec(anc['ra_deg'], anc['dec_deg'])
            label = f'RA {anc["ra_deg"]:.4f} Dec {anc["dec_deg"]:+.4f}'
            span = 0.0
        else:
            l0, b0 = DS.TARGETS[anc['target']]
            M0 = mosaic.frame_M(l0, b0)
            label = f'大致对着 {anc["target"]} (l={l0:.2f}, b={b0:.2f})'
            span = 180.0        # 滚转可能是任意角，全扫
        print(f'锚点帧 {key0}：{label}')

        if span > 0:
            # 只知大致方向：滚转扫描 + 图像互相关，把误差从十几度压到 0.1°
            Ms, diag = RG.coarse_lock(M0, det, Cc, roll_span=span,
                                      roll_step=3.0)
            print(f'  -> 互相关粗定完成（峰值 {diag["peak"]:.0f}）')
        else:
            Gc, nhit, th = attitude.solve_roll(M0, Cb, det)
            Ms = Gc @ M0
            print(f'  滚转扫描：命中 {nhit}/{len(det)}  滚转 {th:+.2f}°')
            Gr, n2, off = attitude.solve_global_offset(Ms, Cb, det, span=1.0,
                                                      step=0.25, tol=6.0)
            if n2 > nhit:
                print(f'  指向微调 dl={off[0]:+.2f} db={off[1]:+.2f} '
                      f'roll={off[2]:+.2f}  命中 {n2}/{len(det)}')
                Ms = Gr @ Ms
        # 换算成相对姿态形式，后面统一走 calibrate_all
        Rinv = np.linalg.inv(R[key0])
        att0 = {k: Ms @ Rinv @ R[k] for k in keys if k in R}
    elif link:
        # 从已标定的 A 组继承绝对姿态
        rawA = json.load(open(os.path.join(CF.RESULT_DIR, 'att_A.json')))
        attA = rawA.get('frames', rawA)
        Mabs = {}
        best_n = {}
        for p in link:
            a, b, Rl = p['a'], p['b'], np.array(p['R'])
            if b not in attA:
                continue
            if a not in best_n or p['n'] > best_n[a][0]:
                best_n[a] = (p['n'], b, Rl)
        for a, (_n, b, Rl) in best_n.items():
            Mabs[a] = np.array(attA[b]) @ Rl.T          # M_new = M_old @ R^T
        print(f'继承绝对姿态：{len(Mabs)}/{len(keys)} 帧')
        # 组内补齐
        adj = {}
        for p in inner:
            adj.setdefault(p['a'], []).append((p['b'], np.array(p['R'])))
            adj.setdefault(p['b'], []).append((p['a'], np.array(p['R']).T))
        for seed in list(Mabs):
            st = [(seed, Mabs[seed])]
            while st:
                a, Ma = st.pop()
                for b, Rab in adj.get(a, []):
                    if b in Mabs:
                        continue
                    Mabs[b] = Ma @ Rab
                    st.append((b, Mabs[b]))
        att0 = {k: v for k, v in Mabs.items()}
    else:
        # 完全盲解：不认识银河、不知道任何方向，只用亮星角距 + 星表
        key0 = max((k for k in keys if k in R), key=lambda k: len(fr[k]))
        o = np.argsort(fr[key0][:, 2])
        uv = fr[key0][:, :2].astype(float)[o][:1500]
        print(f'完全盲解：用星点最多的帧 {key0}（{len(fr[key0])} 个点源）')
        Ms, diag = SL.blind_attitude(uv, n0, G)
        if Ms is None:
            raise RuntimeError('盲解失败：亮星太少，或星表亮星不全')
        Rinv = np.linalg.inv(R[key0])
        att0 = {k: Ms @ Rinv @ R[k] for k in keys if k in R}

    print('逐帧精修：')
    out = attitude.calibrate_all(att0, fr, n0, G)
    path = os.path.join(CF.RESULT_DIR, f'att_{name}.json')
    json.dump({'frames': {k: v.tolist() for k, v in out.items()}}, open(path, 'w'))
    print(f'\n完成 {len(out)}/{len(keys)} -> {path}')


if __name__ == '__main__':
    main()
