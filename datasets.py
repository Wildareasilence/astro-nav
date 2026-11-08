"""数据集定义：文件名 -> 帧编号。

换数据只需在这里加一项，脚本不用改。
"""

PHOTO = 'D:/Program/deepseek/deepgame/12/tmp/picture/1'

# A 组：27 帧，覆盖全天（银道圈 + 上下两圈 + 两个银极）
A = {
    'out': 'framesA.npz',
    'frames': {},
}
for i, n in enumerate(range(36, 43)):
    A['frames'][f'r0_{i:02d}'] = f'scr{n:05d}.png'
for i, n in enumerate(range(87, 96)):
    A['frames'][f'rP_{i:02d}'] = f'scr{n:05d}.png'
for i, n in enumerate(range(53, 60)):
    A['frames'][f'rM_{i:02d}'] = f'scr{n:05d}.png'
A['frames']['poleU_0'] = 'scr00081.png'
A['frames']['poleU_1'] = 'scr00082.png'
A['frames']['poleD_0'] = 'scr00083.png'
A['frames']['poleD_1'] = 'scr00084.png'

# B 组：15 帧，银道带，每帧滚转不同（00101/00102/00114 为拍废删除）
B = {'out': 'framesB.npz', 'frames': {}}
for n in (100, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117):
    B['frames'][f'd{n:03d}'] = f'scr{n:05d}.png'

# C 组：新位置的全天照片。
#   'dir'    照片所在目录（不填则用 config.PHOTO_DIR）
#   'frames' 填 None 表示自动扫描该目录下所有 PNG，帧编号 = 文件名去掉扩展名。
#            也可以照 A/B 的写法手工列出 {帧编号: 文件名}。
C = {'out': 'framesC.npz',
     'dir': 'D:/Program/deepseek/deepgame/12/tmp/picture/3',
     'frames': None,
     # 绝对定向只需要「某一帧大致对着哪里」。两种写法：
     #   1) 认得天体（推荐，现场就能给）：
     #        {'frame': 'scr00203', 'target': 'gc'}     gc/ngp/sgp/lmc/smc/m31/cygnus
     #   2) 有精确赤经赤纬（校准用）：{'frame': ..., 'ra_deg': ..., 'dec_deg': ...}
     'anchor': None,
     }

# 可用的「大致对着某处」锚点：银道坐标。人眼能认出来的天体都用得上。
TARGETS = {
    'gc':   (0.0, 0.0),        # 银心（人马座方向，银河最亮最宽处）
    'ngp':  (0.0, 90.0),       # 北银极（银河成环，天顶很空）
    'sgp':  (0.0, -90.0),      # 南银极
    'lmc':  (280.47, -32.89),  # 大麦云
    'smc':  (302.81, -44.33),  # 小麦云
    'm31':  (121.17, -21.57),  # 仙女座大星系
    'cygnus': (80.0, 0.0),     # 天鹅座方向（北十字，银河亮带）
}

# D 组：新位置，另一套拍摄布局（银道带 18 + 上带 9 + 两银极各 2 + 下带 10）
D = {'out': 'framesD.npz',
     'dir': 'D:/Program/deepseek/deepgame/12/tmp/picture/3',
     'frames': {}}
for i, n in enumerate(range(118, 136)):
    D['frames'][f'b_{i:02d}'] = f'scr{n:05d}.png'
for i, n in enumerate(range(136, 145)):
    D['frames'][f'u_{i:02d}'] = f'scr{n:05d}.png'
for i, n in enumerate(range(147, 157)):
    D['frames'][f'l_{i:02d}'] = f'scr{n:05d}.png'
D['frames']['pA_0'] = 'A银极0°.png'
D['frames']['pA_1'] = 'A银极90°.png'
D['frames']['pB_0'] = 'B银极0度.png'
D['frames']['pB_1'] = 'B银极90度.png'
# b_00（scr00118）大致对着银心（有滚转），从这里逐渐向右扫出整条银道带
D['anchor'] = {'frame': 'b_00', 'target': 'gc'}

DATASETS = {'A': A, 'B': B, 'C': C, 'D': D}

# B 组通过银道圈帧继承 A 组的绝对姿态（两组的重叠帧）
# C 组是新位置，彼此不重叠，所以为 None
LINK_TO = {'A': ['r0_00', 'r0_01', 'r0_02', 'r0_03', 'r0_04', 'r0_05', 'r0_06'],
           'B': None, 'C': None, 'D': None}


def frame_list(name):
    """返回数据集 name 的 [(png 绝对路径, 帧编号), ...]。"""
    import glob
    import os
    d = DATASETS[name]
    pdir = d.get('dir') or PHOTO
    frames = d['frames']
    if frames is None:                       # 自动扫描
        files = sorted(glob.glob(os.path.join(pdir, '*.png'))
                       + glob.glob(os.path.join(pdir, '*.PNG')))
        frames = {os.path.splitext(os.path.basename(f))[0]: os.path.basename(f)
                  for f in files}
    return [(os.path.join(pdir, png), key) for key, png in sorted(frames.items())]
