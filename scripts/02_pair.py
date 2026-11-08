"""步骤 2：帧间配对（三角形边长匹配，旋转不变）。

    python scripts/02_pair.py A
输出 results/pairs_<name>.json
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets as DS
import config as CF
from src import prep, pairmatch as PM


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'A'
    d = DS.DATASETS[name]
    fr = prep.load_frames(d['out'])
    keys = sorted(fr)
    cams = {}
    for k in keys:
        o = np.argsort(fr[k][:, 2])          # 星等升序 = 由亮到暗
        cams[k] = prep.pix2cam(fr[k][:, :2].astype(float))[o]
    print(f'{len(keys)} 帧')

    res = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            n, R, med = PM.try_match(cams[a], cams[b], nimg=40)
            if n >= 8 and R is not None:
                res.append(dict(a=a, b=b, n=int(n), res_deg=float(med), R=R.tolist()))
                print(f'  {a:>8}-{b:<8} 共同 {n:4d}  残差 {med*60:5.2f}\'', flush=True)

    # B 组还要与 A 组的银道圈帧对接，以继承绝对姿态
    link = DS.LINK_TO.get(name)
    if link:
        fa = prep.load_frames(DS.DATASETS['A']['out'])
        for a in keys:
            for b in link:
                if b not in fa:
                    continue
                ca = cams[a]
                ob = np.argsort(fa[b][:, 2])
                cb = prep.pix2cam(fa[b][:, :2].astype(float))[ob]
                n, R, med = PM.try_match(ca, cb, nimg=40)
                if n >= 8 and R is not None:
                    res.append(dict(a=a, b=b, n=int(n), res_deg=float(med),
                                    R=R.tolist(), grp='link'))
                    print(f'  {a:>8}-{b:<8} [链接] 共同 {n:4d}  残差 {med*60:5.2f}\'',
                          flush=True)

    out = os.path.join(CF.RESULT_DIR, f'pairs_{name}.json')
    json.dump(res, open(out, 'w'))
    print(f'\n共 {len(res)} 对 -> {out}')


if __name__ == '__main__':
    main()
