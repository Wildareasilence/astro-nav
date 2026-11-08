"""核心求解器：全局联合 + 每帧姿态 Kabsch 消去 + Huber 稳健加权。

显式参数传递，无全局状态。物理模型（行向量）：
  视差    s_i(x) = (p_i - x)/|p_i - x|
  距离零点 delta: d_true = 1000/(pi_obs - delta)   [pi_obs = 1000/d_cat]
  光行差  s_ab = normalize(s + beta - (s.beta)s)
  姿态    每帧 R_k 由加权 Kabsch 闭式消去（变量投影法）
观测：W_i = 观测方向的 ICRS 表示（由相机方向经精修姿态 M_k 转出）。
"""
import numpy as np

from . import params as PR


def kabsch_w(S, O, wt):
    B = (S*wt[:, None]).T @ (O*wt[:, None])
    A, _s, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(A @ Vt))
    return A @ np.diag([1., 1., d]) @ Vt


class JointSolver:
    """对应集合 (W, ci, groups) 上拟合 theta=[x, beta, delta]。"""

    def __init__(self, u_cat, d_cat):
        """u_cat: (Ncat,3) 星表方向；d_cat: (Ncat,) 星表距离 pc。
        ci 索引指向这两个数组。"""
        self.u = np.asarray(u_cat, float)
        self.d = np.asarray(d_cat, float)

    # ---------- 模型 ----------
    def model_s(self, ci, x, delta=0.0):
        pi = 1000.0/np.maximum(self.d[ci], 1e-3)
        dt = 1000.0/np.maximum(pi-delta, 1e-3)
        p = self.u[ci]*dt[:, None]
        v = p-x
        return v/np.linalg.norm(v, axis=1, keepdims=True)

    def residual(self, theta, W, ci, groups, wt):
        S = self.model_s(ci, theta[:3], theta[6])
        # 必须 zeros：未参与帧的残差保持 0，否则污染雅可比
        out = np.zeros_like(S)
        for g in groups:
            R = kabsch_w(S[g], W[g], wt[g])
            out[g] = S[g] @ R-W[g]
        return out

    @staticmethod
    def group_rows(groups):
        return np.concatenate(groups)

    # ---------- 求解 ----------
    def solve(self, W, ci, groups, x0, delta0=0.0, fit_delta=True,
              w_prior=None, iters=None, verbose=False):
        iters = PR.SOLVER_ITERS if iters is None else iters
        rows = self.group_rows(groups)
        th = np.array([*x0, 0., 0., 0., float(delta0)])
        wt = np.ones(len(W))
        free = [0, 1, 2, 3, 4, 5] + ([6] if fit_delta else [])
        eps = PR.NORM_SCALE_EPS
        hist = []
        for it in range(iters):
            r = self.residual(th, W, ci, groups, wt)
            a = np.linalg.norm(r[rows], axis=1)
            c = PR.HUBER_K*np.median(a)
            wt = np.zeros(len(W))
            base = np.ones(len(W)) if w_prior is None else w_prior
            wt[rows] = base[rows]*np.minimum(1., c/np.maximum(a, 1e-12))
            J = np.empty((len(W)*3, len(free)))
            for j, k in enumerate(free):
                tp = th.copy(); tp[k] += eps[k]
                tm = th.copy(); tm[k] -= eps[k]
                J[:, j] = ((self.residual(tp, W, ci, groups, wt) -
                            self.residual(tm, W, ci, groups, wt)) /
                           (2*eps[k])).ravel()
            Jw = J*np.repeat(wt, 3)[:, None]
            rw = (r*wt[:, None]).ravel()
            dth, *_ = np.linalg.lstsq(Jw, -rw, rcond=None)
            for j, k in enumerate(free):
                th[k] += dth[j]
            th[6] = float(np.clip(th[6], -0.05, 0.05)) if fit_delta else th[6]
            hist.append(th[:3].copy())
            if verbose:
                print(f'      it{it}: |x|={np.linalg.norm(th[:3]):.4f} pc '
                      f'δ={th[6]:+.5f} |dθ|={np.linalg.norm(dth):.2e}',
                      flush=True)
            if np.linalg.norm(dth) < 1e-9:
                break
        return th, wt, hist
