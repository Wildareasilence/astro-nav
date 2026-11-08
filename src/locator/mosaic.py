"""帧提取与帧间拼接：照片 -> 刚体全天图（不依赖绝对指向）。"""
import glob
import os
import math
import time
import json
import numpy as np

from . import params as PR
from .. import prep, pairmatch as PM, mosaic as ANMosaic  # astro-nav


def extract_frames(photo_dir, names=None):
    """照片 -> {key: (N,3) 星点 x,y,mag}。names 为空则扫描全部 png。"""
    if names is None:
        files = sorted(glob.glob(os.path.join(photo_dir, '*.png')))
        names = [os.path.basename(f)[:-4] for f in files]
        files = names
    else:
        files = names
    out = {}
    for k in files:
        p = k if os.path.isabs(k) else os.path.join(photo_dir, k + '.png')
        out[k] = prep.extract_one(p, thr=PR.EXTRACT_THR,
                                  maxstars=PR.EXTRACT_MAXSTARS)
    return out


def build_mosaic(frames, save_dir=None, verbose=True):
    """帧间配对 + 递推，返回 (R, diag)。

    R[k]：相机 k -> seed 相机的相对旋转。约定 w = cam @ R[k].T。
    只依赖重叠星点，与观察者位置无关（同一位置的所有帧一致）。
    """
    keys = sorted(frames)
    cams = {}
    for k in keys:
        o = np.argsort(frames[k][:, 2])
        cams[k] = prep.pix2cam(frames[k][:, :2].astype(float))[o]
    pairs = []
    t0 = time.time()
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            n, R, med = PM.try_match(cams[a], cams[b], nimg=PR.PAIR_NIMG)
            if n >= PR.PAIR_MIN_COMMON and R is not None:
                pairs.append(dict(a=a, b=b, n=int(n),
                                  res_deg=float(med), R=R.tolist()))
    R, seen, order = ANMosaic.propagate(pairs, keys)
    errs = [rot_err(R[p['a']] @ np.array(p['R']), R[p['b']])
            for p in pairs if p['a'] in R and p['b'] in R]
    diag = dict(n_pairs=len(pairs), connected=len(R), n_frames=len(keys),
                closure_med_deg=float(np.median(errs)),
                closure_max_deg=float(max(errs)),
                seconds=time.time()-t0)
    if verbose:
        print(f'  mosaic: {diag["n_pairs"]} 对, 连通 {diag["connected"]}/'
              f'{diag["n_frames"]}, 闭合残差 中位 '
              f'{diag["closure_med_deg"]:.5f}° max {diag["closure_max_deg"]:.4f}°',
              flush=True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        json.dump({k: v.tolist() for k, v in R.items()},
                  open(os.path.join(save_dir, 'R.json'), 'w'))
        json.dump(pairs, open(os.path.join(save_dir, 'pairs.json'), 'w'))
    if errs and max(errs) > 1.0:
        print(f'  WARNING: 闭合残差 max {max(errs):.2f}° > 1° —— '
              f'帧列表可能混入了不同位置/不同批次的照片！'
              f'请显式指定帧名列表。', flush=True)
    return R, diag


def rot_err(A, B):
    return math_deg(math.acos(float(np.clip((np.trace(A.T @ B)-1)/2, -1, 1))))


def math_deg(x):
    return x*180/np.pi
