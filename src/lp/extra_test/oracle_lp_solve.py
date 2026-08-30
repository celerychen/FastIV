#!/usr/bin/env python3
"""Cross-language oracle for fiv_lp_solve.

Runs the REAL PDLP driver (pdlp.py in the or-tools port) on exactly the same
problems as src/lp/test_lp_solve.c, and prints one line per problem:

    ORACLE <name> <status_int> <primal_obj> <dual_obj>

<status_int> uses the same codes as the C enum:
    0 optimal, 1 primal_infeasible, 2 dual_infeasible, 3 iteration_limit, 4 time_limit

compare_oracle.py reads these lines (and the C test's ORACLE lines) and asserts
the two implementations agree -- giving a genuine oracle check against the
algorithm we are porting, not just a self-consistency check.
"""

import sys
import math

import os

# Directory holding the reference pdlp.py we are porting. Override with the
# FIV_PDLP_DIR environment variable; the default is the author's local checkout.
PROJECT_DIR = os.environ.get("FIV_PDLP_DIR",
                             "/Users/celerychen2024/OpenSource/or/or-tools/pdlp/pdlp")
sys.path.insert(0, PROJECT_DIR)

import torch
import pdlp  # the real PDLP driver we are porting

# pdlp.py builds some auxiliary tensors from Python float literals (e.g. the
# Farkas ray in the n==0 trivial-infeasible branch); with torch's default
# float32 those end up float32 while q_orig is float64 and the dot product
# raises. PDLP is meant to run in float64, so pin the default dtype.
torch.set_default_dtype(torch.float64)

STATUS_INT = {
    "optimal": 0,
    "primal_infeasible": 1,
    "dual_infeasible": 2,
    "iteration_limit": 3,
    "time_limit": 4,
}

INF = float("inf")


def solve_problem(name, n, m1, m2, c, G, h, A, b, l, u):
    """G is m1 x n, A is m2 x n (inequality Gx>=h, equality Ax=b, box l<=x<=u)."""
    def vec(values, length):
        if values is None:
            return torch.zeros((length,), dtype=torch.float64)
        return torch.tensor(values, dtype=torch.float64)

    def mat(rows_list, r, c):
        if rows_list is None:
            return torch.zeros((r, c), dtype=torch.float64)
        return torch.tensor(rows_list, dtype=torch.float64)

    c_t = vec(c, n)
    G_t = mat(G, m1, n)
    h_t = vec(h, m1)
    A_t = mat(A, m2, n)
    b_t = vec(b, m2)
    l_t = vec(l, n)
    u_t = vec(u, n)

    try:
        x, y, status, info = pdlp.solve(
            c_t, G_t, h_t, A_t, b_t, l_t, u_t,
            iteration_limit=20000, time_sec_limit=INF,
            eps_tol=1e-4,
        )
    except Exception as exc:  # surface but keep going
        print(f"ORACLE {name} -1 0 0   # ERROR: {exc}")
        return

    si = STATUS_INT.get(status, -1)
    po = float(info.get("primal_obj", 0.0))
    do = float(info.get("dual_obj", 0.0))
    print(f"ORACLE {name} {si} {po:.12g} {do:.12g}")


# ---- Problems: identical to src/lp/test_lp_solve.c -------------------------
# P-A: min -x1-2x2 s.t. x1+x2<=1.3, x in [0,1]^2  -> x=(0.3,1), obj=-2.3
solve_problem("P-A", 2, 1, 0,
    c=[-1, -2], G=[[-1, -1]], h=[-1.3], A=None, b=[], l=[0, 0], u=[1, 1])

# P-B: min x1+2x2 s.t. x1+x2>=1, x in [0,1]^2 -> x=(1,0), obj=1
solve_problem("P-B", 2, 1, 0,
    c=[1, 2], G=[[1, 1]], h=[1], A=None, b=[], l=[0, 0], u=[1, 1])

# P-G: min -x1 s.t. x1>=0, x1+x2=1, x in [0,1]^2 -> x=(1,0), obj=-1
solve_problem("P-G", 2, 1, 1,
    c=[-1, 0], G=[[1, 0]], h=[0], A=[[1, 1]], b=[1], l=[0, 0], u=[1, 1])

# P-F: min x0+x1+x2 s.t. x0>=0, x1>=0, x2=1, free box -> x=(0,0,1), obj=1
solve_problem("P-F", 3, 2, 1,
    c=[1, 1, 1], G=[[1, 0, 0], [0, 1, 0]], h=[0, 0],
    A=[[0, 0, 1]], b=[1], l=[-INF, -INF, -INF], u=[INF, INF, INF])

# P-C: trivial n==0, q=[h=3, b=0] infeasible -> primal_infeasible
solve_problem("P-C", 0, 1, 1,
    c=[], G=None, h=[3], A=None, b=[0], l=[], u=[])

# P-D: trivial m==0 bounded, min x1+x2 over [0,1]^2 -> x=(0,0), obj=0
solve_problem("P-D", 2, 0, 0,
    c=[1, 1], G=None, h=[], A=None, b=[], l=[0, 0], u=[1, 1])

# P-E: trivial m==0 unbounded -> dual_infeasible
solve_problem("P-E", 2, 0, 0,
    c=[-1, -1], G=None, h=[], A=None, b=[], l=[0, 0], u=[INF, INF])

# P-I: non-trivial primal infeasible: x0>=1 AND x0<=0 -> primal_infeasible
solve_problem("P-I", 2, 2, 0,
    c=[0, 0], G=[[1, 0], [-1, 0]], h=[1, 0], A=None, b=[],
    l=[-INF, -INF], u=[INF, INF])

# P-J: non-trivial dual infeasible (primal unbounded): min -x0 s.t. x0>=0 -> dual_infeasible
solve_problem("P-J", 2, 1, 0,
    c=[-1, 0], G=[[1, 0]], h=[0], A=None, b=[], l=[0, -INF], u=[INF, INF])
