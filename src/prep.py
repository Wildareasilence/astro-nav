"""批量星点提取：PNG -> (x, y, mag) 表。"""
import os
import numpy as np

import config as CF
from src import photometry as P


def pix2cam(uv):
    """像素坐标 -> 相机系单位矢量。z 朝前，x 向右，y 向下（左手系，需全局翻 x）。"""
    x = (uv[:, 0] - CF.CX) / CF.FOC
    y = (uv[:, 1] - CF.CY) / CF.FOC
    v = np.stack([x, y, np.ones(len(x))], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def extract_one(path, thr=8.0, maxstars=30000):
    """返回 (N,3) 数组：x, y, mag。mag 越负越亮。"""
    lum, _ = P.load_lum(path)
    bg = P.block_median_bg(lum, 32)
    res = lum - bg
    st, _ = P.detect(lum, thr=thr, maxstars=maxstars)
    if len(st) == 0:
        return np.zeros((0, 3), np.float32)
    # 紧凑度过滤：去掉星云结构
    comp = np.zeros(len(st))
    for i, (x, y, _f) in enumerate(st):
        xi, yi = int(round(x)), int(round(y))
        if xi < 2 or yi < 2 or xi > CF.IMG_W - 3 or yi > CF.IMG_H - 3:
            continue
        comp[i] = (np.clip(res[yi - 1:yi + 2, xi - 1:xi + 2], 0, None).sum()
                   / max(np.clip(res[yi - 2:yi + 3, xi - 2:xi + 3], 0, None).sum(), 1e-6))
    st = st[comp > 0.62]
    mag = -2.5 * np.log10(np.maximum(st[:, 2], 1e-6))
    return np.stack([st[:, 0], st[:, 1], mag], 1).astype(np.float32)


def extract(entries, out_name, thr=8.0, maxstars=30000):
    """entries: [(png_path, key), ...]；写出 FRAME_DIR/out_name，返回 dict。"""
    out = {}
    for path, key in entries:
        if not os.path.exists(path):
            print(f'  缺 {path}')
            continue
        a = extract_one(path, thr=thr, maxstars=maxstars)
        out[key] = a
        print(f'  {os.path.basename(path):>16} -> {key:<10} {len(a):6d} 点源', flush=True)
    np.savez_compressed(os.path.join(CF.FRAME_DIR, out_name), **out)
    print(f'共 {len(out)} 帧 -> {out_name}')
    return out


def load_frames(name):
    return dict(np.load(os.path.join(CF.FRAME_DIR, name)))
