"""步骤 4：位置解算（三个模型比较）。

    python scripts/04_position.py A
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets as DS
import config as CF
from src import prep, solver


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'A'
    d = DS.DATASETS[name]
    fr = prep.load_frames(d['out'])
    raw = json.load(open(os.path.join(CF.RESULT_DIR, f'att_{name}.json')))
    att = raw.get('frames', raw)
    cat = CF.load_catalog()

    rows, Pw, _ = solver.build_rows(att, fr, cat)
    n = sum(len(r[0]) for r in rows)
    print(f'数据集 {name}：{len(rows)} 帧，匹配星 {n}\n')

    res = {}
    for mode, label in [('x', '只有位置 x'), ('v', '只有光行差 v'), ('both', '两者')]:
        p, cov, c2 = solver.fit(rows, Pw, mode)
        print(f'--- {label} ---')
        x, v = solver.report(p, cov, c2, n, mode)
        res[mode] = dict(x=x.tolist(), v=v.tolist(), chi2n=float(c2))

    out = os.path.join(CF.RESULT_DIR, f'posfit_{name}.json')
    json.dump(res, open(out, 'w'))
    print(f'\n-> {out}')
    print('\n判据：χ²/N 最小者为准。若"只解位置"给出的方向正好是反太阳向点，\n'
          '      那多半是光行差被误当视差。')


if __name__ == '__main__':
    main()
