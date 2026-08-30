"""
Compare the C port (test_lp_solve13.c) ORACLE13 lines against the real torch
pdlp.py oracle (oracle_lp_solve13.py).

Validation per problem (the project's mandated status/x/obj consistency check
against tests/test_pdlp.py):
  * status: C must lie in the problem's expected status set (the C test already
    gates this via its exit code). For definite-status problems (no
    iteration_limit in the expected set) C.status must EQUAL torch.status. For
    the "maybe-infeasible" problems P7/P8 (iteration_limit allowed) both C and
    torch just need to be in the expected set.
  * objective: for OPTIMAL problems, C.primal_obj must match torch.primal_obj
    within 1e-3 relative tolerance.
  * solution: for problems with a unique analytic optimum (exp_x given), torch.x
    must match exp_x within x_tol (C.x is already checked in the C test), which
    proves both solvers landed on the same vertex.

Exits non-zero on any mismatch so it can gate the Makefile `oracle13` target.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp13_problems import build_problems


def parse_oracle(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("ORACLE13 "):
                continue
            parts = line.split()
            name = parts[1]
            status = int(parts[2])
            primal_obj = float(parts[3])
            dual_obj = float(parts[4])
            x = [float(v) for v in parts[5:]]
            out[name] = (status, primal_obj, dual_obj, x)
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: compare_oracle13.py <c_oracle.txt> <py_oracle.txt>")
        sys.exit(2)

    problems = {p["name"]: p for p in build_problems()}
    c_data = parse_oracle(sys.argv[1])
    py_data = parse_oracle(sys.argv[2])

    g_fail = 0
    g_pass = 0

    for name in sorted(problems.keys()):
        p = problems[name]
        if name not in c_data:
            print(f"  FAIL {name}: missing C oracle line")
            g_fail += 1
            continue
        if name not in py_data:
            print(f"  FAIL {name}: missing torch oracle line")
            g_fail += 1
            continue

        c_status, c_obj, c_dobj, c_x = c_data[name]
        t_status, t_obj, t_dobj, t_x = py_data[name]
        exp_status = set(p["exp_status"])

        # --- status ---
        flexible = 3 in exp_status  # P7 / P8 allow iteration_limit
        if flexible:
            ok = (c_status in exp_status) and (t_status in exp_status)
            detail = "flexible(infeasible/limit)"
        else:
            ok = (c_status == t_status) and (c_status in exp_status)
            detail = "definite"
        if ok:
            g_pass += 1
            print(f"  OK   {name}: status={c_status} (torch={t_status}) [{detail}]")
        else:
            g_fail += 1
            print(f"  FAIL {name}: status C={c_status} torch={t_status} "
                  f"exp={sorted(exp_status)}")

        # --- objective (only meaningful when optimal) ---
        if c_status == 0 and t_status == 0:
            rel = abs(c_obj - t_obj) / (1.0 + abs(t_obj))
            if rel <= 1e-3:
                g_pass += 1
                print(f"  OK   {name}: primal_obj C={c_obj:.6g} torch={t_obj:.6g}")
            else:
                g_fail += 1
                print(f"  FAIL {name}: primal_obj C={c_obj:.6g} torch={t_obj:.6g} "
                      f"rel_err={rel:.2e}")

        # --- solution (unique optimum problems) ---
        if p["exp_x"] is not None and c_status == 0 and t_status == 0:
            tol = max(p["x_tol"], 0.02)
            ok_x = True
            for i, ev in enumerate(p["exp_x"]):
                if i >= len(t_x):
                    ok_x = False
                    break
                if abs(t_x[i] - ev) > tol * (1.0 + abs(ev)):
                    ok_x = False
                    break
            if ok_x:
                g_pass += 1
                print(f"  OK   {name}: torch x matches expected vertex (tol={tol:g})")
            else:
                g_fail += 1
                print(f"  FAIL {name}: torch x={t_x} expected={p['exp_x']}")

    print(f"\noracle13 comparison: {g_pass} passed, {g_fail} failed")
    sys.exit(0 if g_fail == 0 else 1)


if __name__ == "__main__":
    main()
