"""步骤 7：stellar_locator 全自动盲定位（统一网格拍摄）。

    python scripts/07_locate.py E

数据集需在 datasets.py 里定义 'grid'（bands/poles/anchor）。
拍摄要求见 README「定位器」一节：每圈 6 帧 × 60° 银经、首帧 l≈0、
带水平滚转≈0、两银极各 2 张互成 90°。

输出：results/locator_<name>/locator_result.json（坐标、诊断、评分）。
truth 仅作校准评分用，现场数据不填即为纯盲解。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets as DS
import config as CF
from src.locator.pose import GridSpec
from src.locator.pipeline import locate


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'E'
    d = DS.DATASETS[name]
    g = d.get('grid')
    if not g:
        raise SystemExit(
            f"数据集 {name} 没有 'grid' 定义。定位器要求统一网格拍摄：\n"
            "  bands: [(银纬, 起始银经, 步进, [帧名...]), ...]\n"
            "  poles: {'north': [...], 'south': [...]}\n"
            "  anchor: 首帧帧名（大致对银心 l≈0）")

    grid = GridSpec(bands=g['bands'], poles=g['poles'])
    keys = sorted({k for _, _, _, ks in g['bands'] for k in ks}
                  | set(g['poles']['north']) | set(g['poles']['south']))

    out_dir = os.path.join(CF.RESULT_DIR, f'locator_{name}')
    res = locate(photo_dir=d.get('dir') or CF.PHOTO_DIR,
                 frames=keys, grid=grid,
                 anchor_key=g.get('anchor') or keys[0],
                 out_dir=out_dir,
                 truth=d.get('truth'))
    print()
    print(res['summary'])
    if res.get('score'):
        s = res['score']
        print(f"校准评分: 距离差 {s['dist_ly']:+.5f} ly  角差 {s['ang_arcsec']:.2f}\"  "
              f"综合 {s['total_ly']:.5f} ly ({s['total_ly']*63241.08:.0f} AU)")
    print('结果已写入', out_dir)


if __name__ == '__main__':
    main()
