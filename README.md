# astro-nav — 全天星场盲定位工具

从一批**指向未知**的广角星场照片 + 三维星表，反解：

- 每帧的绝对姿态（相机 → ICRS）
- 观察者的三维位置 `x`（AU）
- 附带：观察者速度造成的恒星光行差 `v`（km/s）

整套方法在真实渲染数据上跑通的实测精度是 **位置 ~100 AU**（0.002 光年），
瓶颈是像素尺度（165″/px），不是算法。

详细推导、判据、踩坑记录见 **[docs/pipeline.md](docs/pipeline.md)**。

---

## 目录结构

```
astro-nav/
├── config.py              路径 / 光学常数 / 银道矩阵（改这里换数据集）
├── datasets.py            文件名 -> 帧编号的映射
├── src/
│   ├── photometry.py      读图、背景、点源检测、质心
│   ├── sky.py             Gaia CSV -> 三维星表；观测方程
│   ├── prep.py            批量提取星点 -> npz
│   ├── pairmatch.py       三角形边长匹配（旋转不变）
│   ├── mosaic.py          拼接递推 + 银道锚点 + 闭合检验
│   ├── starlock.py        完全盲解：亮星角距配对 + RANSAC + Kabsch（默认路径）
│   ├── register.py        图像互相关粗定（认得天体时用，±15° 捕获）
│   ├── attitude.py        逐帧姿态标定（3+1 参数）、滚转/全局偏移搜索
│   ├── solver.py          位置 + 光行差联合解算
│   └── diagnose.py        三个诊断（距离分层 / 滚转对 / 历元扫描）
├── scripts/               按序号执行
│   ├── 00_download_catalog.py
│   ├── 01_extract.py
│   ├── 02_pair.py
│   ├── 03_attitude.py
│   ├── 04_position.py
│   └── 05_diagnose.py
├── data/
│   ├── catalog/           Gaia CSV（已附 288,800 颗）
│   └── frames/            framesA/B.npz（已附）；新数据存 framesC/D.npz
├── results/               姿态矩阵、配对、位置解、全天图
└── docs/pipeline.md       完整技术文档
```

## 依赖

Python 3.10+，`numpy`、`Pillow`。无 scipy。

## 快速开始

星表和星点表都已附好，**直接从第 4 步跑就能复现结果**：

```bash
python scripts/04_position.py A     # 27 帧全天覆盖 -> 位置解
python scripts/04_position.py B     # 15 帧银道带（不同滚转）-> 位置解
```

从头跑（需要原始照片，路径在 `config.PHOTO_DIR`）：

```bash
python scripts/00_download_catalog.py   # 首次；已有 data/catalog 则跳过
python scripts/01_extract.py A          # PNG -> framesA.npz
python scripts/02_pair.py    A          # -> results/pairs_A.json
python scripts/03_attitude.py A         # -> results/att_A.json
python scripts/04_position.py A         # -> results/posfit_A.json
python scripts/05_diagnose.py A all     # 三个诊断
```

B 组要先有 A 组的 `att_A.json`（它的绝对姿态是从 A 组继承的）：


```bash
python scripts/01_extract.py B
python scripts/02_pair.py    B
python scripts/03_attitude.py B
python scripts/04_position.py B
```

## 换一组新位置的照片

**照片放这里**（已在 `datasets.py` 里配好，随便命名，自动扫描全部 PNG）：

```
D:/Program/deepseek/deepgame/12/tmp/picture/3/      <- 临时目录，新位置的照片
```

放进去后：

```bash
python scripts/01_extract.py  C     # 自动扫描该目录
python scripts/02_pair.py     C
python scripts/03_attitude.py C     # 需要一个已知中心，见下
python scripts/04_position.py C
```

`03_attitude.py` **默认完全盲解**：不需要赤经赤纬，也不需要认银河，只要照片和星表。

做法（`src/starlock.py`，标准星敏感器算法）：

1. 取该帧最亮的 80 颗，要求彼此 >0.8°（避开双星与混合像元）；
2. 用「最亮几颗之间的大角距」当种子，在星表亮星里找同长的配对 → 得到一个旋转；
3. 拿这个旋转投影星表亮星，数落在实拍星点上的个数（随机期望个位数）；
4. 最优解做 Kabsch 精修，交给 `calibrate_frame`。

实测（A 组，27 帧里抽 9 帧，覆盖银道圈、上下两圈、两个银极）：

| 帧 | 误差 | 用时 |
|---|---:|---:|
| r0_00 / r0_02 / r0_04 / r0_06 | 10″ / 11″ / 13″ / 6″ | 16–26 s |
| rP_00 / rP_04 | 7″ / 9″ | 18–26 s |
| rM_03 / poleU_0 / poleD_0 | 3″ / 9″ / 14″ | 21–27 s |

**9/9 全部锁定**，中位 9″。整组 27 帧跑完后与已知真值差 2″，位置 112.7 AU ——
和用精确赤经赤纬锚点得到的结果完全一致。

### 但这个「盲」有个前提，必须说清楚

星图识别默认「拍到的图案 ≈ 星表里的图案」。这只在你离星表原点（太阳）很近时成立。
位移 `x` 会让距离 `L` 的星移动 `x/L` 弧度，而能用来识别的亮星中位距离只有几百 pc：

| 离太阳 | 做法 | 实测误差 |
|---:|---|---:|
| 0 | 默认（全星表，容差 120″） | **4″** |
| 1 ly | 只用 d>200 pc 的星，容差 300″ | 58–67″ |
| 3 ly | d>300 pc，容差 600″ | 129″ |
| 10 ly | d>500 pc，容差 1500″ | 425″（0.12°） |
| 30 ly | d>800 pc，容差 2400″ | 841″（0.23°） |
| 100 ly | — | **错解** |

1–30 ly 这几档是「手动选对档」的结果；`starlock.blind_auto` 会自动试几档，
但只在 1 ly 以内可靠（更远时各档得分趋同，挑不出来）。
超出约 1 ly，正确的做法是把位置也一起搜——见 `docs/pipeline.md` §12。

### 如果你认得出天体，可以走捷径

误差 ±15° 都能自动压到 0.1°，比盲解快得多：

```python
C['anchor'] = {'frame': 'scr00203', 'target': 'gc'}
```

| target | 含义 | 现场怎么认 |
|---|---|---|
| `gc` | 银心 | 银河最宽最亮、有暗尘带横切的那一团 |
| `ngp` / `sgp` | 南/北银极 | 银河成环贴住画幅四边，中央很空 |
| `lmc` / `smc` | 大/小麦云 | 南天两团独立的云雾 |
| `m31` | 仙女座大星系 | 一个椭圆雾斑 |

做法是图像互相关：把星表渲染成图，与实拍星点图做 FFT，滚转全扫 360°、平移一次解出。
实测起手偏 16.7° 仍能收敛到 0.05°，偏 26.9° 失败。

有精确赤经赤纬时也可以写 `{'ra_deg': 266.49, 'dec_deg': -28.82}`（赤经用**度**）。

其它注意事项：

- 图片尺寸必须和 `config.IMG_W/IMG_H` 一致，`01_extract.py` 会检查并提示。
- 短轴视场变了就改 `config.VFOV_DEG`；焦距、像素尺度会自动重算。
- 拍摄要求不变：全天覆盖，相邻帧重叠 ≥ 20%，**每帧故意转不同角度**（这样质心系统差会互相抵消）。



| 步骤 | 输入 → 输出 | 成功判据 |
|---|---|---|
| 1 提取 | PNG → `(x, y, mag)` 每帧 1–2 万 | 紧凑度过滤后仍 >1 万点源 |
| 2 配对 | 星点表 → `(共同星数, R, 残差)` | 共同星 ≥ 8，残差 ≤ 1′ |
| 3 姿态 | 配对 → 每帧 3×3 旋转 | 每帧内点 1000+，中位残差 ≤ 0.2 px |
| 4 位置 | 姿态 + 星表 → `x`, `v` | 三个模型的 χ²/N 比较，取最小者 |
| 5 诊断 | 同上 | |x| 不随距离上限变化 ⇒ 真视差 |

## 结果（本仓库数据）

```
A 组 27 帧（全天覆盖）       B 组 15 帧（银道带，滚转跨度 128°）
  只解位置    |x| = 642 AU      只解位置    |x| = 669 AU
  只解光行差  v = 19.1 km/s      只解光行差  v = 18.8 km/s
  两者        |x| = 111 AU      两者        |x| = 168 AU
              v = 17.5 km/s                 v = 16.6 km/s
```

真值：观察者在地球附近，|x| ≤ 0.07 AU。

反演出的光行差速度指向 **l≈60°, b≈+26°**，
与太阳本动（19.5 km/s，l=57°, b=+22°）吻合——这是一项独立的物理检验。

## 四条必须遵守的约定

1. **配对旋转**：`cam_a @ R = cam_b`，即 `R = M_aᵀ M_b`。
2. **拼接递推**：`R_b = R_a @ R`。写成 `Rᵀ @ R_a` 只有种子帧正确。
3. **手性**：相机系是左手系（x 右、y 下、z 前），必须翻 x，
   且 `M = A @ FLIP @ R`（FLIP 在 R **左边**）。
4. **"银道水平" ≠ 滚转 0**：手工调平通常差 0.5–1°，在 2560 px 上就是十几像素，
   足以让所有像素级匹配无声失败。必须用 `solve_global_offset` 搜出来。

## 已知限制

- 单星质心误差 ~0.2 px（33″），受渲染器亚像素精度限制，换质心算法无改善（实测）。
- 位置精度 ∝ 像素尺度。45° 视场预计 ~79 AU，30° 预计 ~51 AU。
- 位置与光行差是形状相同的两个偶极，只靠星表距离跨度分开；距离范围窄时会退化。
- 未处理径向畸变（实测该项 <0.04 px，可忽略）和星表视差零点。
