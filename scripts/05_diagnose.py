"""步骤 5：诊断（判断系统差属于哪一类）。

    python scripts/05_diagnose.py A all
    python scripts/05_diagnose.py A stratify
    python scripts/05_diagnose.py A roll  poleU_0 poleU_1
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets as DS
import config as CF
from src import prep, diagnose


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'A'
    what = sys.argv[2] if len(sys.argv) > 2 else 'all'
    d = DS.DATASETS[name]
    fr = prep.load_frames(d['out'])
    raw = json.load(open(os.path.join(CF.RESULT_DIR, f'att_{name}.json')))
    att = raw.get('frames', raw)
    cat = CF.load_catalog()

    if what in ('all', 'stratify'):
        diagnose.stratify(att, fr, cat, tag=name)
    if what in ('all', 'epoch'):
        print()
        diagnose.epoch_scan(att, fr, cat)
    if what in ('all', 'roll'):
        print()
        args = sys.argv[3:]
        if len(args) >= 2:
            diagnose.roll_pair(att, fr, cat, args[0], args[1])
        else:
            for a, b in [('poleU_0', 'poleU_1'), ('poleD_0', 'poleD_1')]:
                if a in att and b in att:
                    diagnose.roll_pair(att, fr, cat, a, b)


if __name__ == '__main__':
    main()
