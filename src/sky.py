"""
Gaia DR3 近星子样本 -> 三维星表 + 观测方程。

单位：长度 AU，时间 yr，c = 63241.077 AU/yr，1 km/s = 4.7404704 AU/yr。
坐标系：ICRS 日心系，历元由 Gaia DR3 给出（J2016.0）。
"""
import os
import numpy as np

AU_PER_PC = 206264.80624709636
C_AU_YR = 63241.07708426628
KMS_AUYR = 4.7404704463
MAS_RAD = 4.84813681109536e-9
ARCSEC = 4.84813681109536e-6
ARCSEC_RAD = 4.84813681109536e-6
DEG = np.pi / 180.0
AU_PER_LY = 63241.07708426628
EPOCH0 = 2016.0


def norm_rows(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def unit_from_radec(ra_deg, dec_deg):
    ra = np.asarray(ra_deg) * DEG
    dec = np.asarray(dec_deg) * DEG
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


def radec_from_unit(u):
    u = norm_rows(np.atleast_2d(u))
    ra = np.arctan2(u[:, 1], u[:, 0]) % (2 * np.pi)
    dec = np.arcsin(np.clip(u[:, 2], -1, 1))
    return ra / DEG, dec / DEG


def load_gaia(glob_paths, verbose=True):
    """读 CSV 分片，返回星表字典（AU, AU/yr）。"""
    import glob as _g
    files = sorted(_g.glob(glob_paths))
    rows = []
    hdr = None
    for f in files:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            head = fh.readline().strip()
            if hdr is None:
                hdr = head.split(',')
            for line in fh:
                parts = line.rstrip('\n').split(',')
                if len(parts) != len(hdr):
                    continue
                rows.append(parts)
    if verbose:
        print(f'  read {len(rows)} rows from {len(files)} files')
    A = np.array(rows, dtype=object)
    col = {name: i for i, name in enumerate(hdr)}

    def f(name):
        v = np.array([x[col[name]] for x in A], dtype=object)
        out = np.full(len(v), np.nan)
        for i, s in enumerate(v):
            if s not in ('', 'null', 'NaN'):
                out[i] = float(s)
        return out

    ra, dec = f('ra'), f('dec')
    plx, plx_err = f('parallax'), f('parallax_error')
    pmra, pmdec = f('pmra'), f('pmdec')
    rv = f('radial_velocity')
    G, bp_rp = f('phot_g_mean_mag'), f('bp_rp')
    ruwe = f('ruwe')

    ok = (np.isfinite(plx) & np.isfinite(pmra) & np.isfinite(pmdec)
          & np.isfinite(G) & np.isfinite(ra) & np.isfinite(dec))
    for k in (ra, dec, plx, plx_err, pmra, pmdec, rv, G, bp_rp, ruwe):
        del k
    ra, dec, plx, plx_err, pmra, pmdec, rv, G, bp_rp, ruwe = (
        a[ok] for a in (ra, dec, plx, plx_err, pmra, pmdec, rv, G, bp_rp, ruwe))

    d_pc = 1000.0 / plx
    u = unit_from_radec(ra, dec)

    # 切向基
    rar, decr = ra * DEG, dec * DEG
    e_ra = np.stack([-np.sin(rar), np.cos(rar), np.zeros_like(rar)], axis=-1)
    e_dec = np.stack([-np.sin(decr) * np.cos(rar),
                      -np.sin(decr) * np.sin(rar),
                      np.cos(decr)], axis=-1)

    has_rv = np.isfinite(rv)
    rv = np.where(has_rv, rv, 0.0)
    vt_ra = 4.7404704463e-3 * pmra * d_pc      # km/s
    vt_dec = 4.7404704463e-3 * pmdec * d_pc
    v_kms = (rv[:, None] * u + vt_ra[:, None] * e_ra + vt_dec[:, None] * e_dec)

    cat = dict(
        u=u, ra=ra, dec=dec,
        d_pc=d_pc, d=d_pc * AU_PER_PC,
        pos=u * (d_pc * AU_PER_PC)[:, None],
        vel=v_kms / KMS_AUYR,
        v_kms=v_kms,
        G=G, bp_rp=bp_rp, ruwe=ruwe,
        plx=plx, s_plx=plx_err,
        has_rv=has_rv,
        s_pmra=np.abs(pmra) * 0 + 0.0,
    )
    # 每颗星的位置误差椭球：径向 = d^2 * sigma_plx / 1000（pc）
    cat['sig_rad_pc'] = d_pc ** 2 * plx_err / 1000.0
    cat['sig_rad'] = cat['sig_rad_pc'] * AU_PER_PC
    # 横向位置误差 0.02 mas（Gaia DR3 亮星典型），质量差的按 G 放大
    sig_pos_mas = 0.02 + 0.02 * 10 ** (0.2 * (G - 17.0))
    cat['sig_pos_mas'] = sig_pos_mas
    cat['sig_tan'] = sig_pos_mas * MAS_RAD * d_pc * AU_PER_PC

    if verbose:
        print(f'  G range {G.min():.1f}..{G.max():.1f}, d {d_pc.min():.2f}..{d_pc.max():.1f} pc')
        print(f'  has RV: {has_rv.sum()}/{len(has_rv)}  '
              f'median sig_rad {np.median(cat["sig_rad"]):.0f} AU')
    return cat


def subset(cat, idx):
    n = len(cat['G'])
    out = {}
    for k, v in cat.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- 观测方程
def light_time_dirs(x, p0, vel, t, t0=EPOCH0, iters=4):
    """从观察者 x 指向恒星的几何方向，含光行时。x: (3,), p0/vel: (N,3)."""
    tem = np.full(len(p0), t)
    for _ in range(iters):
        d = p0 + vel * (tem - t0)[:, None] - x
        L = np.linalg.norm(d, axis=1)
        tem = t - L / C_AU_YR
    d = p0 + vel * (tem - t0)[:, None] - x
    L = np.linalg.norm(d, axis=1)
    return d / L[:, None], L, tem


def aberrate(s, beta):
    b2 = float(beta @ beta)
    if b2 > 0.9:
        beta = np.asarray(beta) * (np.sqrt(0.9) / np.sqrt(b2))
        b2 = 0.9
    if b2 < 1e-24:
        return norm_rows(s + beta - (s @ beta)[:, None] * s)
    g = 1.0 / np.sqrt(1.0 - b2)
    sb = s @ beta
    num = s / g + beta + (g / (g + 1.0)) * sb[:, None] * beta
    return norm_rows(num / (1.0 + sb)[:, None])


def unaberrate(n, beta):
    b2 = float(beta @ beta)
    if b2 > 0.9:
        beta = np.asarray(beta) * (np.sqrt(0.9) / np.sqrt(b2))
        b2 = 0.9
    if b2 < 1e-24:
        return norm_rows(n - beta + (n @ beta)[:, None] * n)
    g = 1.0 / np.sqrt(1.0 - b2)
    nb = n @ beta
    num = n / g - beta + (g / (g + 1.0)) * nb[:, None] * beta
    return norm_rows(num / (1.0 - nb)[:, None])


def doppler_factor(n, beta):
    """n: 视线指向恒星（观察者系近似）。返回 D = nu_obs/nu_emit。"""
    b2 = float(beta @ beta)
    if b2 < 1e-24:
        return np.ones(len(n))
    g = 1.0 / np.sqrt(1.0 - b2)
    return g * (1.0 + n @ beta)


def rot_axis_angle(w):
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = np.asarray(w) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def project(x, cat, beta=None, t=None, epoch0=EPOCH0, beaming=False):
    """返回 (观察者系单位方向 n, 距离 L, 星等修正)。"""
    t = 2016.0 if t is None else t
    s, L, _ = light_time_dirs(x, cat['pos'], cat['vel'], t, epoch0)
    n = s if beta is None else aberrate(s, beta)
    dm = np.zeros(len(n))
    if beaming and beta is not None:
        D = doppler_factor(s, beta)
        dm = -2.5 * np.log10(D ** 4)
    return n, L, dm


def sky_position(x, cat, idx, t=2026.0, beta=None):
    """给定 x，返回指定星在天空中的方向、距离、视星等。"""
    s, L, _ = light_time_dirs(x, cat['pos'][idx], cat['vel'][idx], t, EPOCH0)
    n = s if beta is None else aberrate(s, beta)
    G = cat['G'][idx] + 5.0 * np.log10(L / (10 * AU_PER_PC))
    if beta is not None:
        G = G - 2.5 * np.log10(doppler_factor(s, beta) ** 4)
    return n, L, G


def app_mag(cat, L, idx=None, dm=0.0):
    """从观察者位置看恒星的视星等。Gaia 的 G 是地球视星等，需换距离模数。"""
    if idx is None:
        d = np.linalg.norm(cat['pos'], axis=1)
        G = cat['G']
    else:
        d = np.linalg.norm(cat['pos'][idx], axis=1)
        G = cat['G'][idx]
    return G + 5.0 * np.log10(np.maximum(L, 1.0) / np.maximum(d, 1.0)) + dm


def project_sub(x, cat, idx, beta=None, t=2016.0, epoch0=EPOCH0):
    """对索引子集做前向投影（不含集束星等修正）。"""
    s, L, _ = light_time_dirs(x, cat['pos'][idx], cat['vel'][idx], t, epoch0)
    n = s if beta is None else aberrate(s, beta)
    return n, L, np.zeros(len(n))
