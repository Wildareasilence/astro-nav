"""顶层流程：照片目录 + 拍摄网格 -> 观察者三维坐标。

用法（最短路径）：
    from stellar_locator.pipeline import locate
    grid = GridSpec(
        bands=[(0., 0., -60., [f'scr{n:05d}' for n in range(183,189)]),
               (40., 0., -60., [f'scr{n:05d}' for n in range(189,195)]),
               (-40., 0., -60., [f'scr{n:05d}' for n in range(197,203)])],
        poles={'north': ['PA1','PA2'], 'south': ['PB1','PB2']})
    res = locate(photo_dir='.../picture/3', grid=grid)
    print(res['summary'])

全程不预知位置。粗解只用于启动；对应集合随解迭代重建。
"""
import json
import math
import os
import time
import numpy as np

from . import params as PR
from . import mosaic as MO
from . import pose as PS
from . import coarse as CO
from . import correspond as CR
from .solver import JointSolver
from .. import attitude                     # astro-nav（calib 用）


def load_catalog(path=None):
    path = PR.CATALOG_NPZ if path is None else path
    z = np.load(path)
    u = z['u'].astype(float)
    G = z['G'].astype(float)
    d = z['dpc'].astype(float)
    logd = np.log10(np.maximum(d, 1e-3))
    return u, G, d, logd


def refine_loop(frames, M_frames, x0, P, G, D, logd, solver,
                rounds=None, tol=None, verbose=True):
    """迭代重建对应 + 全局联合（核心。粗解只用于启动）。"""
    rounds = PR.REFINE_ROUNDS if rounds is None else rounds
    tol = PR.REFINE_TOL_LY if tol is None else tol
    x = np.array(x0, float)
    hist = []
    W = ci = groups = None
    for rnd in range(rounds):
        W, ci, groups, per = CR.build_correspondences(
            frames, M_frames, x, P, G, D, logd)
        th, wt, _ = solver.solve(W, ci, groups, x)
        xn = th[:3]
        dx = float(np.linalg.norm(xn-x)*PR.PC_LY)
        if verbose:
            print(f'  refine 轮{rnd}: 对应 {len(W)} ({len(groups)}帧) '
                  f'x={xn.round(5)} |x|={np.linalg.norm(xn)*PR.PC_LY:.4f} ly '
                  f'Δ={dx:.6f} ly δ={th[6]:+.5f} mas', flush=True)
        hist.append(xn.copy())
        x = xn
        if dx < tol:
            break
    return x, dict(W=W, ci=ci, groups=groups, per=per, hist=hist,
                   theta=th, wt=wt)


def per_frame_solve(W, ci, groups, solver, x_ref, delta=0.0):
    """22 帧各自独立解（一致性检验）。

    fit_delta 必须为 False：δ 在单帧内与 x 径向简并，
    逐帧拟合会把噪声灌进 |x| 散布（实测 σ 虚增 6 倍）。
    """
    X = []
    for g in groups:
        th, _wt, _h = solver.solve(W[g], ci[g], [np.arange(len(g))],
                                   x_ref, delta0=delta, fit_delta=False,
                                   iters=14)
        X.append(th[:3])
    return np.array(X)


def split_cross(W, ci, groups, solver, x0, seed=None):
    """半帧交叉验证。"""
    rng = np.random.default_rng(PR.RANDOM_SEED if seed is None else seed)
    od = rng.permutation(len(groups))
    outs = []
    for hv in (od[:len(od)//2], od[len(od)//2:]):
        gr = [groups[j] for j in hv]
        th, _wt, _h = solver.solve(W, ci, gr, x0)
        outs.append(th[:3])
    return outs


def convergence(W, ci, groups, solver, x_final, offsets=(20.,)):
    """从远处初值出发的收敛域检验。"""
    starts = [np.zeros(3)]
    for o in offsets:
        for ax in range(3):
            v = np.zeros(3); v[ax] = o
            starts.append(x_final+v)
            starts.append(x_final-v)
    errs = []
    for s in starts:
        th, _wt, _h = solver.solve(W, ci, groups, s)
        errs.append(float(np.linalg.norm(th[:3]-x_final)*PR.PC_LY))
    return errs


def _make_thetas(idx, ii, jj, P):
    """角距闭包：s_i(x) 的两两夹角（纯几何，不含姿态/光行差）。"""
    def thetas(xv):
        d = P[idx]-xv
        s = d/np.linalg.norm(d, axis=1, keepdims=True)
        return np.arccos(np.clip((s[ii]*s[jj]).sum(1), -1, 1))
    return thetas


def angdist_check(key, frames, M_frames, x, P, G, D, logd):
    """纯角距闭环（不用姿态矩阵）：160 星分层 + 全对角距 Gauss-Newton。

    从 x=0 出发也应收敛到同一解 -> 排除姿态链路的循环论证。
    注意：对应仍须用 calib 精修后的姿态建立（0.588° 的 Wahba 残差
    远超 3px 容差）；但解 x 只用星间角距，姿态不进入拟合。
    """
    fr = frames[key]
    det = fr[:, :2].astype(float)[
        np.argsort(fr[:, 2])[:PR.DETS_PER_FRAME]]
    sk, sg = CR.skeleton(P, G, logd, x)
    acc, _sc, _nin, _med = attitude.calib_frame_scale(
        M_frames[key], sk, sg, det[:4000], gmax=PR.CALIB_GMAX, iters=10)
    M = M_frames[key] if acc is None else M_frames[key] @ acc
    pred, qi = CR.project_frame(x, M, P, G, logd)
    nb = 8
    bk = {}
    for j, (u, v) in enumerate(det):
        bk.setdefault((int(u//nb), int(v//nb)), []).append(j)
    oj, cidx = [], []
    for i, (u, v) in enumerate(pred):
        cu, cv = int(u//nb), int(v//nb)
        best, bj = 9., -1
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                for j in bk.get((cu+ax, cv+ay), ()):
                    d2 = (det[j, 0]-u)**2+(det[j, 1]-v)**2
                    if d2 < best: best, bj = d2, j
        if bj >= 0:
            oj.append(bj); cidx.append(qi[i])
    oj = np.array(oj, np.int64); cidx = np.array(cidx, np.int64)
    from .. import prep as _prep
    ow = _prep.pix2cam(det[oj])
    dd = D[cidx]
    order = np.argsort(dd)
    sel = order[np.linspace(0, len(order)-1,
                            min(160, len(order))).astype(int)]
    w = ow[sel]; idx = cidx[sel]
    ii, jj = np.triu_indices(len(w), 1)
    th_obs = np.arccos(np.clip((w[ii]*w[jj]).sum(1), -1, 1))
    thetas = _make_thetas(idx, ii, jj, P)

    # 多起点阻尼 GN：从 x=0 出发的线性化在 |x|~20 pc 时可能失效
    # （实测 rms 6.8° 处停滞），从粗解方向多个尺度出发取 rms 最小
    best = None
    for start in (np.zeros(3), x*0.3, x*0.6, x.copy()):
        xx = np.array(start, float)
        rms = float(np.sqrt(np.mean((thetas(xx)-th_obs)**2)))
        for _ in range(60):
            t = thetas(xx)
            r = t-th_obs
            J = np.zeros((len(t), 3)); h = 0.02
            for k in range(3):
                dx = xx.copy(); dx[k] += h
                J[:, k] = (thetas(dx)-t)/h
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            s = 1.0
            for _a in range(8):
                trial = xx-s*step
                rms1 = float(np.sqrt(np.mean((thetas(trial)-th_obs)**2)))
                if rms1 < rms:
                    xx, rms = trial, rms1
                    break
                s *= 0.5
            else:
                break
            if np.linalg.norm(s*step) < 1e-7:
                break
        if best is None or rms < best[1]:
            best = (xx, rms)
    x, rms = best
    rms_arcsec = float(np.degrees(rms)*3600)
    return x, rms_arcsec


def sky_coords(x):
    r = float(np.linalg.norm(x)); u = x/r
    ra = math.degrees(math.atan2(u[1], u[0])) % 360
    dec = math.degrees(math.asin(float(np.clip(u[2], -1, 1))))
    lb = PR.AN.GAL @ x
    gu = PR.AN.GAL @ u
    l = math.degrees(math.atan2(gu[1], gu[0])) % 360
    b = math.degrees(math.asin(float(np.clip(gu[2], -1, 1))))
    return dict(dist_pc=r, dist_ly=r*PR.PC_LY, ra_deg=ra, dec_deg=dec,
                gal_l=l, gal_b=b,
                sun_dir_ra_deg=(ra+180) % 360, sun_dir_dec_deg=-dec,
                sun_apparent_mag=PR.SUN_ABS_MAG+5*math.log10(max(r, 1e-9)/10),
                gal_xyz_ly=(lb*PR.PC_LY).tolist(),
                xyz_pc=x.tolist())


def locate(photo_dir, grid, anchor_key=None, frames=None, catalog=None,
           coarse_start=None, out_dir=None, truth=None, verbose=True):
    """主入口。返回 dict（含 summary）。

    grid       : pose.GridSpec 拍摄网格
    anchor_key : 外部银经零点锚（"该帧大致对 l=0"），缺省取第一个带帧
    frames     : {key: (N,3)}，缺省自动提取 photo_dir 全部 png
    truth      : dict(dist_ly, ra_deg, dec_deg)，仅供事后评分
    """
    t_start = time.time()
    truth = PR.TRUTH if truth is None else truth
    if frames is None:
        if verbose: print('=== 1 提取星点 ===', flush=True)
        frames = MO.extract_frames(photo_dir)
        if verbose:
            print(f'  警告：未显式指定帧列表，扫描到 {len(frames)} 张。'
                  f'目录里若混有其他批次照片，请改传 frames=帧名列表。',
                  flush=True)
    elif not hasattr(frames, 'items'):
        # 传的是帧名列表（可含扩展名） -> 提取这些帧
        if verbose: print(f'=== 1 提取星点（显式 {len(frames)} 帧）===',
                          flush=True)
        frames = MO.extract_frames(photo_dir, names=list(frames))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    keys = sorted(frames)
    if anchor_key is None:
        anchor_key = grid.bands[0][3][0]

    if verbose: print('=== 2 帧间拼接 ===', flush=True)
    R_rel, mos_diag = MO.build_mosaic(frames, save_dir=out_dir,
                                      verbose=verbose)

    if verbose: print('=== 3 Wahba 定向（外部锚 dl，不扫描）===', flush=True)
    Q, M_frames, pose_diag = PS.solve_wahba(R_rel, grid, verbose=verbose)
    anchor_diag = PS.check_anchor(M_frames, anchor_key,
                                  l_expect=grid.bands[0][1])
    if verbose:
        print(f'  锚帧 {anchor_key} 对齐后 l={anchor_diag["l"]:.2f}° '
              f'b={anchor_diag["b"]:.2f}°', flush=True)

    u, G, D, logd = load_catalog(catalog)
    Pc = u*D[:, None]                 # 星表三维位置（pc）
    solver = JointSolver(u, D)
    frame0 = grid.bands[0][3][0]

    if verbose: print('=== 4 多尺度粗搜 x ===', flush=True)
    det0 = frames[frame0][:, :2].astype(float)[
        np.argsort(frames[frame0][:, 2])[:PR.DETS_PER_FRAME]]
    x_coarse, coarse_diag = CO.multi_scale_search(
        det0, M_frames[frame0], Pc, G, logd, x0=coarse_start,
        verbose=verbose)

    if verbose: print('=== 5 迭代重建对应 + 全局联合 ===', flush=True)
    x, ref = refine_loop(frames, M_frames, x_coarse, Pc, G, D, logd,
                         solver, verbose=verbose)

    if verbose: print('=== 6 验证 ===', flush=True)
    W, ci, groups = ref['W'], ref['ci'], ref['groups']
    Xf = per_frame_solve(W, ci, groups, solver, x, delta=ref['theta'][6])
    per_sigma = Xf.std(0)
    cross = split_cross(W, ci, groups, solver, np.zeros(3))
    cross_err = [float(np.linalg.norm(c-x)*PR.PC_LY) for c in cross]
    conv = convergence(W, ci, groups, solver, x)
    # 纯角距闭环（用锚帧）
    xad, ad_rms = angdist_check(frame0, frames, M_frames, x, Pc, G, D,
                                logd)
    ad_err = float(np.linalg.norm(xad-x)*PR.PC_LY)
    if verbose:
        print(f'  22帧独立 σ={per_sigma.round(5)} pc '
              f'(|x| σ={np.linalg.norm(Xf,axis=1).std()*PR.PC_LY:.5f} ly)',
              flush=True)
        print(f'  半帧交叉差 {cross_err[0]:.5f}/{cross_err[1]:.5f} ly',
              flush=True)
        print(f'  收敛域最大偏差 {max(conv):.6f} ly', flush=True)
        print(f'  纯角距闭环 {np.linalg.norm(xad)*PR.PC_LY:.4f} ly '
              f'差 {ad_err:.4f} ly (角距rms {ad_rms:.0f}")', flush=True)

    # 真值评分（可选）
    score = None
    if truth:
        u = -x/np.linalg.norm(x)
        ra = math.degrees(math.atan2(u[1], u[0])) % 360
        dec = math.degrees(math.asin(float(np.clip(u[2], -1, 1))))
        dra = ((ra-truth['ra_deg']+180) % 360-180)*3600
        ddec = (dec-truth['dec_deg'])*3600
        ang = math.hypot(dra*math.cos(math.radians(truth['dec_deg'])),
                         ddec)
        dd = np.linalg.norm(x)*PR.PC_LY-truth['dist_ly']
        score = dict(dist_ly=float(dd), ang_arcsec=float(ang),
                     total_ly=float(math.hypot(dd, ang/206264.8 *
                                               np.linalg.norm(x)*PR.PC_LY)))
        if verbose:
            print(f'  真值评分: 距离差 {dd:+.5f} ly  角差 {ang:.2f}"  '
                  f'综合 {score["total_ly"]:.5f} ly '
                  f'({score["total_ly"]*63241.08:.0f} AU)', flush=True)

    sky = sky_coords(x)
    summary = (f'观察者位于距太阳 {sky["dist_ly"]:.3f} 光年处，'
               f'方向 RA {sky["ra_deg"]:.3f}° Dec {sky["dec_deg"]:+.3f}°')
    res = dict(x_pc=x.tolist(), sky=sky, summary=summary,
               anchor=anchor_diag, mosaic=mos_diag, pose=pose_diag,
               coarse=coarse_diag, refine_hist=[h.tolist()
                                                for h in ref['hist']],
               per_frame=ref['per'],
               per_frame_sigma_pc=per_sigma.tolist(),
               cross_err_ly=cross_err, convergence_max_ly=float(max(conv)),
               angdist=dict(x=xad.tolist(), err_ly=ad_err, rms_arcsec=ad_rms),
               delta_mas=float(ref['theta'][6]),
               beta=ref['theta'][3:6].tolist(),
               n_corr=int(len(W)), score=score,
               seconds=time.time()-t_start)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(res, open(os.path.join(out_dir, 'locator_result.json'),
                            'w'), indent=1)
        np.save(os.path.join(out_dir, 'x_located.npy'), x)
    return res
