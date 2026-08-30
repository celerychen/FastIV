"""
Cross-language oracle for the 13 problems: run the REAL torch pdlp.py
(pdlp.solve) on the exact data from lp13_problems.build_problems() and emit
"ORACLE13 <name> <status_int> <primal_obj> <dual_obj> <x...>" lines, matching
the format printed by test_lp_solve13.c. compare_oracle13.py then checks the
C port against this ground truth.

status_int mapping: 0 optimal, 1 primal_infeasible, 2 dual_infeasible,
3 iteration_limit, 4 time_limit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Directory holding the reference pdlp.py we are porting. Override with the
# FIV_PDLP_DIR environment variable; the default is the author's local checkout.
sys.path.insert(0, os.environ.get("FIV_PDLP_DIR",
                                  "/Users/celerychen2024/OpenSource/or/or-tools/pdlp/pdlp"))

import torch
from pdlp import solve

# pdlp.py builds internal tensors with the global default dtype. The C port and
# our inputs are all float64, so force float64 here too; otherwise degenerate
# problems (e.g. P10, n=0) hit a float32-vs-float64 `dot` mismatch inside solve.
torch.set_default_dtype(torch.float64)

from lp13_problems import build_problems

STATUS_INT = {
    "optimal": 0,
    "primal_infeasible": 1,
    "dual_infeasible": 2,
    "iteration_limit": 3,
    "time_limit": 4,
}


def to_tensor(values, *shape):
    if len(values) == 0 and len(shape) > 0:
        return torch.zeros(*shape, dtype=torch.float64)
    return torch.tensor(values, dtype=torch.float64).reshape(*shape)


def main():
    problems = build_problems()
    for p in problems:
        n = p["n"]
        m1 = p["m1"]
        m2 = p["m2"]
        G = to_tensor(p["G"], m1, n)
        h = to_tensor(p["h"], m1)
        A = to_tensor(p["A"], m2, n)
        b = to_tensor(p["b"], m2)
        c = to_tensor(p["c"], n)
        l = to_tensor(p["l"], n)
        u = to_tensor(p["u"], n)

        try:
            x_sol, y_sol, status, info = solve(
                c, G, h, A, b, l, u,
                iteration_limit=10000,   # match C port params
                eps_tol=1e-4,
                verbose=False,
            )
            status_int = STATUS_INT.get(status, -1)
            # Normalize dtypes: pdlp may return x_sol/y_sol in a lower precision
            # (e.g. P10 yields float32) while c is float64 -> coerce before dot.
            x_sol = x_sol.to(torch.float64)
            y_sol = y_sol.to(torch.float64)
            primal_obj = float((c @ x_sol).item())
            dual_obj = float(info.get("dual_obj", 0.0))
        except Exception as exc:  # pragma: no cover
            print(f"ORACLE13 {p['name']} -1 0 0 # ERROR: {exc}")
            continue

        parts = [f"ORACLE13 {p['name']} {status_int} {primal_obj:.12g} {dual_obj:.12g}"]
        for i in range(n):
            parts.append(f"{float(x_sol[i].item()):.12g}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
