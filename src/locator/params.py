"""可调参数集中处。改这里即可换数据集/相机，不必动算法。

已并入 astro-nav：依赖包内相对导入；config 取自 astro-nav 根目录。
"""
import os

import numpy as np
import config as AN                   # astro-nav 根目录的 config.py

# ---------- 相机（默认取 astro-nav config，可覆盖） ----------
IMG_W = AN.IMG_W
IMG_H = AN.IMG_H
FOC = AN.FOC
CX, CY = AN.CX, AN.CY
VFOV_DEG = AN.VFOV_DEG

# ---------- 星表 ----------
# cat.npz 至少含: u (N,3 单位方向 ICRS), G (N 视星等),
#                 dpc (N 距离 pc), bp (N, 可选颜色)
# 深星表（845,673 颗，u/G/dpc/bp）。默认用包内副本；可用环境变量覆盖。
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # astro-nav/src
CATALOG_NPZ = os.environ.get(
    'LOCATOR_CATALOG',
    os.path.join(os.path.dirname(_PKG), 'data', 'catalog', 'cat.npz'))

# ---------- 几何 ----------
PC_LY = 3.261563777
AU_PER_PC = 206264.80624709636
RMAX_PC = 100.0 / PC_LY               # 搜索球半径（30.66 pc）
SUN_ABS_MAG = 4.83                    # 太阳绝对星等（视星等预测用）

# ---------- 星点提取 ----------
EXTRACT_THR = 8.0
EXTRACT_MAXSTARS = 30000
DETS_PER_FRAME = 12000                # 参与匹配的每帧亮星上限

# ---------- 拼接 ----------
PAIR_MIN_COMMON = 8
PAIR_NIMG = 40

# ---------- Wahba 定向 ----------
WAHBA_B_GRID = (15., 20., 25., 30., 35., 40., 45., 50., 60.)
# dl（银经零点）与绕银极旋转严格简并，必须由外部锚点给定，不扫描。
WAHBA_DL = 0.0

# ---------- 多尺度粗搜 ----------
COARSE_MLIM = 10.0
COARSE_CATALOG_GMAX = 11.5            # 渲染用星表上限（留 1 等余量）
COARSE_LEVELS = ((32, 8.0, 2), (16, 4.0, 2), (8, 2.0, 2),
                 (4, 1.0, 2), (2, 0.5, 2))
# 注意 levels[0] 只用于 L0 全空间网格的 bin；细化从 levels[1] 开始
# （bin 32 的分辨率 ~1.5°/格，峰背比仅 1.03，不能作细化级）
COARSE_REFINE_FROM = 1                # levels 中第一个细化级下标
COARSE_N_KEEP = 12
COARSE_GRID_PC = 8.0                  # L0 步长

# ---------- 对应与精修 ----------
CORRESPOND_RAD_PX = 3.0
CALIB_GMAX = 20.0                     # calib_frame_scale 的 gmax
SKELETON_GMAX = 9.5
SKELETON_NMAX = 45000
MODEL_MLIM = 10.0
DIST_MIN_PC = 2.0                     # 剔除太近/占位距离星
DIST_MAX_PC = 15000.
SOLVER_ITERS = 9
REFINE_ROUNDS = 5
REFINE_TOL_LY = 1e-5                  # 迭代重建对应的收敛判据

# ---------- 求解器 ----------
HUBER_K = 1.5
NORM_SCALE_EPS = np.array([1e-4] * 3 + [1e-7] * 3 + [1e-6])

# ---------- 验证 ----------
RANDOM_SEED = 31

# ---------- 真值（可选，仅供评分；不含则跳过） ----------
TRUTH = None                          # dict(dist_ly=..., ra_deg=..., dec_deg=...)
