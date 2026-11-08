"""24 帧两两配对：纯旋转不变的三角形匹配，不依赖任何布局假设。"""
import itertools
import numpy as np

DEG = np.pi / 180
FOC = 720.0 / np.tan(22.5 * DEG)


def pix2cam(uv):
    x = (uv[:, 0] - 1280.0) / FOC
    y = (uv[:, 1] - 720.0) / FOC
    v = np.stack([x, y, np.ones(len(x))], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def tris(u):
    n = len(u)
    D = np.arccos(np.clip(u @ u.T, -1, 1))
    t = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
    s = np.sort(np.stack([D[t[:, 0], t[:, 1]], D[t[:, 0], t[:, 2]],
                          D[t[:, 1], t[:, 2]]], 1), 1)
    # 丢掉过小的三角形（形状退化，易误配）
    keep = (s[:, 0] > 3 * DEG) & (s[:, 2] < 150 * DEG)
    return s[keep], t[keep]


def kabsch(p, q):
    B = p.T @ q
    U, sv, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def try_match(camA, camB, nimg=45, tol=0.25 * DEG, inl=0.06 * DEG, maxtry=400):
    ia = np.argsort(-camA[:, 2])[:nimg] if camA.shape[1] > 2 else slice(None)
    A = camA[:nimg]
    B = camB[:nimg]
    sA, tA = tris(A)
    sB, tB = tris(B)
    if len(sA) == 0 or len(sB) == 0:
        return 0, None, 99.0
    order = np.argsort(sB[:, 0])
    sb0 = sB[order, 0]
    cand_i, cand_j = [], []
    for k in range(len(sA)):
        lo = np.searchsorted(sb0, sA[k, 0] - tol, 'left')
        hi = np.searchsorted(sb0, sA[k, 0] + tol, 'right')
        if hi <= lo:
            continue
        c = order[lo:hi]
        ok = (np.abs(sB[c, 1] - sA[k, 1]) < tol) & (np.abs(sB[c, 2] - sA[k, 2]) < tol)
        if ok.any():
            cand_i.append(np.full(ok.sum(), k))
            cand_j.append(c[ok])
    if not cand_i:
        return 0, None, 99.0
    cand_i = np.concatenate(cand_i)
    cand_j = np.concatenate(cand_j)
    step = max(1, len(cand_i) // maxtry)
    best = (0, None, 99.0)
    for k in range(0, len(cand_i), step):
        R = kabsch(A[tA[cand_i[k]]], B[tB[cand_j[k]]])
        pred = A @ R
        cos = pred @ B.T
        ang = np.arccos(np.clip(cos.max(1), -1, 1))
        n = int((ang < inl).sum())
        if n > best[0]:
            keep = ang < inl
            if keep.sum() >= 3:
                R = kabsch(A[keep], B[np.argmax(cos[keep], 1)])
                pred = A @ R
                ang = np.arccos(np.clip((pred @ B.T).max(1), -1, 1))
                n = int((ang < inl).sum())
            best = (n, R, float(np.median(ang[ang < inl]) / DEG) if n else 99.0)
    return best
