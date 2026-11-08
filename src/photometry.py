"""从广角星空照片提取星点：背景建模、阈值、亚像素质心、相对星等。"""
import numpy as np
from PIL import Image


def load_lum(path):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).astype(np.float32)
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return lum, a


def block_median_bg(lum, block=32):
    """块中值背景，双线性放大。抗星点与大面积星云。"""
    H, W = lum.shape
    ny, nx = H // block, W // block
    b = lum[:ny * block, :nx * block].reshape(ny, block, nx, block)
    med = np.median(b, axis=(1, 3))
    # 用 np.interp 逐行/逐列放大
    yi = (np.arange(H) + 0.5) / block - 0.5
    xi = (np.arange(W) + 0.5) / block - 0.5
    y0 = np.clip(np.floor(yi).astype(int), 0, ny - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    bg = (med[y0][:, x0] * (1 - fy) * (1 - fx) + med[y0][:, x1] * (1 - fy) * fx
          + med[y1][:, x0] * fy * (1 - fx) + med[y1][:, x1] * fy * fx)
    return bg


def detect(lum, block=32, thr=6.0, npix=3, maxstars=4000):
    bg = block_median_bg(lum, block)
    res = lum - bg
    sig = 1.4826 * np.median(np.abs(res - np.median(res)))
    sig = max(sig, 1e-3)
    sigmap = np.zeros_like(lum) + sig
    # 噪声随亮度变化：用局部块的标准差
    H, W = lum.shape
    bb = 64
    ny, nx = H // bb, W // bb
    rb = res[:ny * bb, :nx * bb].reshape(ny, bb, nx, bb)
    loc = 1.4826 * np.median(np.abs(rb - np.median(rb, axis=(1, 3),
                                                  keepdims=True)), axis=(1, 3))
    loc = np.maximum(loc, sig * 0.8)
    yi = (np.arange(H) + 0.5) / bb - 0.5
    xi = (np.arange(W) + 0.5) / bb - 0.5
    y0 = np.clip(np.floor(yi).astype(int), 0, ny - 1)
    x0 = np.clip(np.floor(xi).astype(int), 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    sigmap = (loc[y0][:, x0] * (1 - fy) * (1 - fx) + loc[y0][:, x1] * (1 - fy) * fx
              + loc[y1][:, x0] * fy * (1 - fx) + loc[y1][:, x1] * fy * fx)

    snr = res / sigmap
    # 局部极大
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(snr, 1, mode='constant', constant_values=-1e9)
    win = sliding_window_view(pad, (3, 3))
    ismax = snr >= win.reshape(snr.shape[0], snr.shape[1], 9).max(axis=2) - 1e-9
    cand = np.where(ismax & (snr > thr))
    ys, xs = cand
    if len(ys) == 0:
        return np.zeros((0, 3)), sig

    # 5x5 通量 + 3x3 质心
    out = []
    for y, x in zip(ys, xs):
        if y < 3 or x < 3 or y > H - 4 or x > W - 4:
            continue
        patch = res[y - 2:y + 3, x - 2:x + 3]
        w = np.clip(patch - 0.5 * sigmap[y, x], 0, None)
        flux = w.sum()
        if flux <= 0:
            continue
        dw = w.sum()
        yy = (w * (np.arange(-2, 3)[:, None])).sum() / dw
        xx = (w * (np.arange(-2, 3)[None, :])).sum() / dw
        out.append((x + xx, y + yy, flux))
    out = np.array(out)
    if len(out) > maxstars:
        out = out[np.argsort(-out[:, 2])[:maxstars]]
    return out, sig


def to_mag(flux):
    f = np.maximum(flux, 1e-6)
    return -2.5 * np.log10(f)


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        'D:/Program/deepseek/deepgame/12/tmp/picture/1/0,0.png'
    lum, _ = load_lum(p)
    st, sig = detect(lum)
    m = to_mag(st[:, 2])
    print(f'{p}: {len(st)} stars, bg noise {sig:.3f}, mag range {m.min():.1f}..{m.max():.1f}')
    print('  亮度分布 (每 0.5 等):')
    h, e = np.histogram(m, bins=np.arange(np.floor(m.min()), np.ceil(m.max()) + 0.5, 0.5))
    for i in range(len(h)):
        print(f'    {e[i]:6.1f}  {h[i]:5d}')
