#!/usr/bin/env python3
"""Independent pure-Python oracle for the PDLP PDHG core (M3).

Reimplements the four functions straight from pdlp.py (no shared code with the
C port, no numpy) on the SAME small problem used by src/lp/test_lp_pdhg.c, and
prints the reference numbers. Used to confirm the C test's independent dense
reference matches pdlp.py's true behavior (third-party discrimination).

Run:  python3 oracle_lp_pdhg.py
"""
import math

EPS_ZERO = 1e-12
EPS_TOL = 1e-4

# ---- shared problem (matches test_lp_pdhg.c) ----
N, M, M1 = 3, 3, 2
c = [1.0, -2.0, 0.5]
l = [-1.0, float("-inf"), 0.0]
u = [2.0, float("inf"), 1.0]
q = [3.0, 1.0, 3.0]          # h0=3, h1=1, b=3
# K stacked [G; A], M x N
K = [[1.0, 0.0, 1.0],
     [0.0, 1.0, -1.0],
     [1.0, 1.0, 1.0]]


def matvec(K, x):
    """K (m x n) @ x (n) -> m-vector."""
    return [sum(K[r][cc] * x[cc] for cc in range(len(x))) for r in range(len(K))]


def matvec_t(K, y):
    """K.T (n x m) @ y (m) -> n-vector."""
    n = len(K[0])
    m = len(K)
    return [sum(K[r][cc] * y[r] for r in range(m)) for cc in range(n)]


def clip(v, lo_vec, hi_vec):
    out = []
    for i in range(len(v)):
        val = v[i]
        lo = lo_vec[i]
        hi = hi_vec[i]
        if math.isfinite(lo) and val < lo:
            val = lo
        if math.isfinite(hi) and val > hi:
            val = hi
        out.append(val)
    return out


def adaptive_step(x, y, K, c, l, u, q, m1, w, eta_hat, k):
    w = max(w, EPS_ZERO)
    eta = max(eta_hat, EPS_ZERO)
    kp1 = k + 1
    fac1 = 1.0 if k == 0 else 1.0 - kp1 ** -0.3
    fac2 = 1.0 + kp1 ** -0.6
    while True:
        KTy = matvec_t(K, y)
        grad_x = [c[i] - KTy[i] for i in range(N)]
        x_p = clip([x[i] - (eta / w) * grad_x[i] for i in range(N)], l, u)
        t = [2.0 * x_p[i] - x[i] for i in range(N)]
        Kt = matvec(K, t)
        grad_y = [q[i] - Kt[i] for i in range(M)]
        y_raw = [y[i] + (eta * w) * grad_y[i] for i in range(M)]
        y_p = y_raw[:]
        for i in range(m1):
            if y_p[i] < 0.0:
                y_p[i] = 0.0
        dx = [x_p[i] - x[i] for i in range(N)]
        dy = [y_p[i] - y[i] for i in range(M)]
        dx_sq = sum(d * d for d in dx)
        dy_sq = sum(d * d for d in dy)
        num = w * dx_sq + dy_sq / w
        Kdx = matvec(K, dx)
        dyKdx = sum(dy[i] * Kdx[i] for i in range(M))
        denom = 2.0 * abs(dyKdx)
        bar_eta = float("inf") if denom <= EPS_ZERO else num / denom
        eta_p = min(fac1 * bar_eta, fac2 * eta)
        eta_p = max(eta_p, EPS_ZERO)
        if eta <= bar_eta:
            return x_p, y_p, eta, eta_p
        eta = eta_p


def compute_lambda_box(x, g, lower, upper, eps_tol):
    lam = [0.0] * len(x)
    for i in range(len(x)):
        xi, gi, li, ui = x[i], g[i], lower[i], upper[i]
        at_l = math.isfinite(li) and (xi <= li + eps_tol)
        at_u = math.isfinite(ui) and (xi >= ui - eps_tol)
        if at_l and at_u:
            dl = abs(xi - li)
            du = abs(ui - xi)
            at_l = dl <= du
            at_u = du < dl
        if at_l:
            lam[i] = gi if gi > 0.0 else 0.0
        elif at_u:
            lam[i] = gi if gi < 0.0 else 0.0
    return lam


def kkt_error_sq(K, c, l, u, q, m1, x, y, w, eps_tol):
    w = max(w, EPS_ZERO)
    Kx = matvec(K, x)
    KTy = matvec_t(K, y)
    r_eq_sq = 0.0
    r_ineq_sq = 0.0
    for i in range(M):
        if i < m1:
            r = q[i] - Kx[i]
            if r > 0.0:
                r_ineq_sq += r * r
        else:
            r = Kx[i] - q[i]
            r_eq_sq += r * r
    term1 = w * w * (r_eq_sq + r_ineq_sq)
    g = [c[i] - KTy[i] for i in range(N)]
    lam = compute_lambda_box(x, g, l, u, eps_tol)
    rs_sq = sum((g[i] - lam[i]) ** 2 for i in range(N))
    term2 = (1.0 / (w * w)) * rs_sq
    l_term = 0.0
    u_term = 0.0
    for i in range(N):
        lp = lam[i] if lam[i] > 0.0 else 0.0
        ln = -lam[i] if lam[i] < 0.0 else 0.0
        if math.isfinite(l[i]):
            l_term += l[i] * lp
        if math.isfinite(u[i]):
            u_term += u[i] * ln
    qy = sum(q[i] * y[i] for i in range(M))
    cx = sum(c[i] * x[i] for i in range(N))
    scalar = qy + l_term - u_term - cx
    term3 = scalar * scalar
    return term1 + term2 + term3


def primal_weight_update(x_new, y_new, x_old, y_old, w_old, smoothing):
    dx = math.sqrt(sum((x_new[i] - x_old[i]) ** 2 for i in range(N)))
    dy = math.sqrt(sum((y_new[i] - y_old[i]) ** 2 for i in range(M)))
    if dx > EPS_ZERO and dy > EPS_ZERO:
        return (dy / dx) ** smoothing * w_old ** (1.0 - smoothing)
    return w_old


def main():
    x = [0.5, 0.0, 0.5]
    y = [0.1, -0.2, 0.3]
    w = 1.0
    eta_hat = 0.5
    xp, yp, eta_used, eta_hat_next = adaptive_step(x, y, K, c, l, u, q, M1, w, eta_hat, 0)
    print("adaptive_step (k=0):")
    print("  x_p      =", ["%.15g" % v for v in xp])
    print("  y_p      =", ["%.15g" % v for v in yp])
    print("  eta_used =", repr(eta_used))
    print("  eta_hat  =", repr(eta_hat_next))
    kkt = kkt_error_sq(K, c, l, u, q, M1, x, y, w, EPS_TOL)
    print("kkt_error_sq =", repr(kkt))

    xl = [-1.0, 0.0, 1.0]
    gl = [2.0, -3.0, 4.0]
    lam = compute_lambda_box(xl, gl, l, u, EPS_TOL)
    print("lambda_box (x=[-1,0,1], g=[2,-3,4]) =", ["%.15g" % v for v in lam])

    x_new = [0.6, 0.1, 0.4]
    y_new = [0.2, -0.1, 0.5]
    x_old = [0.5, 0.0, 0.5]
    y_old = [0.1, -0.2, 0.3]
    pw = primal_weight_update(x_new, y_new, x_old, y_old, 1.0, 0.5)
    print("primal_weight =", repr(pw))


if __name__ == "__main__":
    main()
