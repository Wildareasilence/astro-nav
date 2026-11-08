"""步骤 1：提取星点。

    python scripts/01_extract.py A
    python scripts/01_extract.py C        # 新位置，自动扫描 datasets.C['dir']

会自动核对图片尺寸与 config 里的 IMG_W/IMG_H；不一致时直接停下并给出改法，
因为焦距、主点、像素尺度全部依赖它。
"""
import os
import sys
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets as DS
import config as CF
from src import prep


def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(33)
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    return struct.unpack('>II', head[16:24])


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'A'
    d = DS.DATASETS[name]
    entries = DS.frame_list(name)
    if not entries:
        print(f'数据集 {name}：目录里没有 PNG。')
        pdir = d.get('dir') or CF.PHOTO_DIR
        print(f'  期望目录：{pdir}')
        return

    sz = png_size(entries[0][0])
    print(f'数据集 {name}：{len(entries)} 帧，焦距 {CF.FOC:.1f}px，'
          f'像素尺度 {CF.ARCSEC_PX:.1f}″/px')
    if sz and sz != (CF.IMG_W, CF.IMG_H):
        print(f'\n[停止] 图片尺寸是 {sz[0]}×{sz[1]}，config 里写的是 '
              f'{CF.IMG_W}×{CF.IMG_H}。')
        print('       请改 config.py 的 IMG_W/IMG_H（视场角不用动，'
              '焦距会按短轴自动重算）。')
        print('       若这组的短轴视场也变了，同时改 VFOV_DEG。')
        return
    prep.extract(entries, d['out'])
    print(f'  目录：{os.path.dirname(entries[0][0])}')


if __name__ == '__main__':
    main()
