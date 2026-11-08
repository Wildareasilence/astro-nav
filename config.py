"""全局配置：路径、光学常数、银道坐标矩阵。

所有模块都从这里取参数，改这一处即可换数据集。
"""
import os
import glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------- 路径 ----------
# 原始照片目录（只读）。设为空字符串则用 ROOT/data/photos
PHOTO_DIR = r'D:/Program/deepseek/deepgame/12/tmp/picture/1'
# Gaia 星表 CSV 目录
CATALOG_DIR = os.path.join(ROOT, 'data', 'catalog')
# 提取后的星点表
FRAME_DIR = os.path.join(ROOT, 'data', 'frames')
# 结果输出
RESULT_DIR = os.path.join(ROOT, 'results')
for _d in (FRAME_DIR, RESULT_DIR, CATALOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------- 光学 ----------
IMG_W, IMG_H = 2560, 1440            # 单帧像素尺寸
VFOV_DEG = 60.0                      # 短轴（垂直）视场
FOC = (IMG_H / 2) / np.tan(np.radians(VFOV_DEG / 2))   # 焦距，像素
CX, CY = IMG_W / 2.0, IMG_H / 2.0    # 主点
ARCSEC_PX = 180 / np.pi * 3600 / FOC  # 像素尺度（角秒/像素）

# ---------- 天文常数 ----------
AU_PER_PC = 206264.80624709636
C_KMS = 299792.458
GAL_EPOCH = 2026.0                   # 星表传播到的历元

# ICRS -> Galactic（行向量左乘）
GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                [0.4941094279, -0.4448296300, 0.7469822445],
                [-0.8676661490, -0.1980763734, 0.4559837762]])


def catalog_files(which='near'):
    """near: 180 pc 内 G<12 子样本；bright: 全天 G<8.5。"""
    pat = 'part_*.csv' if which == 'near' else 'bb_*.csv'
    files = sorted(glob.glob(os.path.join(CATALOG_DIR, pat)))
    out = []
    for f in files:
        with open(f, encoding='utf-8', errors='ignore') as fh:
            if fh.readline().count(',') > 5:
                out.append(f)
    return out


def load_catalog(verbose=True):
    """读入并合并星表，返回 sky.load_gaia 的字典。"""
    from src import sky
    files = catalog_files('near')
    if not files:
        raise FileNotFoundError(
            f'{CATALOG_DIR} 下没有 part_*.csv。\n'
            '先运行 scripts/00_download_catalog.py，或把 Gaia CSV 放进去。')
    cat = sky.load_gaia(','.join(files) if len(files) == 1 else
                        os.path.join(CATALOG_DIR, 'part_*.csv'), verbose=verbose)
    bright = catalog_files('bright')
    if bright:
        parts = [sky.load_gaia(f, verbose=False) for f in bright]
        n = len(cat['G'])
        for k in cat:
            if isinstance(cat[k], np.ndarray) and cat[k].shape[:1] == (n,):
                cat[k] = np.concatenate([cat[k]] + [q[k] for q in parts])
    if verbose:
        print(f'  星表合计 {len(cat["G"])} 颗')
    return cat
