#!/usr/bin/env python3
"""Compare the C test's ORACLE summary against the real torch pdlp.py oracle.

Usage: compare_oracle.py <c_summary.txt> <py_summary.txt>

Parses lines of the form "ORACLE <name> <status> <primal_obj> <dual_obj>"
from each file and asserts the two implementations agree:
  - status must match exactly (same integer code);
  - for an OPTIMAL result (status 0) primal_obj must match within 1e-3
    relative tolerance (the dual objective is also cross-checked);
  - for infeasible / unbounded / iteration_limit results only the status is
    compared (the oracle does not report a meaningful objective there).

Exits non-zero if any problem disagrees, so it can gate a Makefile target.
"""

import sys

OBJ_TOL = 1e-3


def parse(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("ORACLE "):
                continue
            parts = line.split()
            name = parts[1]
            status = int(parts[2])
            primal = float(parts[3])
            dual = float(parts[4])
            out[name] = (status, primal, dual)
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: compare_oracle.py <c_summary> <py_summary>")
        return 2

    c = parse(sys.argv[1])
    py = parse(sys.argv[2])

    names = list(c.keys())
    fails = 0
    for name in names:
        if name not in py:
            print(f"  FAIL {name}: missing from python oracle")
            fails += 1
            continue
        cs, cpo, cdo = c[name]
        ps, ppo, pdo = py[name]
        if cs != ps:
            print(f"  FAIL {name}: status mismatch C={cs} torch={ps}")
            fails += 1
            continue
        if cs == 0:  # OPTIMAL -> compare objectives too
            denom = 1.0 + abs(ppo) + abs(cpo)
            if abs(cpo - ppo) > OBJ_TOL * denom:
                print(f"  FAIL {name}: primal_obj mismatch C={cpo:.8g} torch={ppo:.8g}")
                fails += 1
                continue
            denom = 1.0 + abs(pdo) + abs(cdo)
            if abs(cdo - pdo) > OBJ_TOL * denom:
                print(f"  FAIL {name}: dual_obj mismatch C={cdo:.8g} torch={pdo:.8g}")
                fails += 1
                continue
        print(f"  OK   {name}: status={cs} (primal_obj C={cpo:.6g} torch={ppo:.6g})")

    print(f"\n=== oracle comparison: {len(names) - fails} passed, {fails} failed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
