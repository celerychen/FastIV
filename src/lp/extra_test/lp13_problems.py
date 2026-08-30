"""
Single source of truth for the 13 LP problems in tests/test_pdlp.py.

Both the C header generator (gen_lp13_header.py) and the torch oracle
(oracle_lp_solve13.py) import build_problems() so the C port and the real
torch pdlp.py are fed byte-identical data. Every problem is transcribed
verbatim from tests/test_pdlp.py (same Gx>=h / Ax=b / l<=x<=u convention).

STATUS CODES (int, shared with fiv_lp_status and oracle output):
    0 optimal
    1 primal_infeasible
    2 dual_infeasible
    3 iteration_limit
    4 time_limit
"""

import torch


def _t(x):
    return x.tolist()


def build_problems():
    problems = []

    def add(name, n, m1, m2, G, h, A, b, c, l, u,
            exp_status, exp_x=None, x_tol=1e-2, obj_tol=1e-3):
        problems.append(dict(
            name=name, n=n, m1=m1, m2=m2,
            G=_t(G) if G.numel() else [],
            h=_t(h),
            A=_t(A) if A.numel() else [],
            b=_t(b),
            c=_t(c), l=_t(l), u=_t(u),
            exp_status=list(exp_status),
            exp_x=(_t(exp_x) if exp_x is not None else None),
            x_tol=float(x_tol), obj_tol=float(obj_tol),
        ))

    # ---- Problem 1: 2D inequality, unique vertex x=[1,0], obj=1 ----
    add("P1", 2, 1, 0,
        torch.tensor([[1.0, 1.0]]), torch.tensor([1.0]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]),
        exp_status=[0], exp_x=torch.tensor([1.0, 0.0]), x_tol=0.01, obj_tol=1e-3)

    # ---- Problem 2: equality x1+2x2=3, x>=0, unique x=[0,1.5], obj=1.5 ----
    add("P2", 2, 0, 1,
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([[1.0, 2.0]]), torch.tensor([3.0]),
        torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]),
        exp_status=[0], exp_x=torch.tensor([0.0, 1.5]), x_tol=0.01, obj_tol=1e-3)

    # ---- Problem 3: 3D multiple inequalities, obj=3 (multiple optima) ----
    add("P3", 3, 3, 0,
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([3.0, 1.0, 1.0]),
        torch.tensor([]).reshape(0, 3), torch.tensor([]),
        torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([10.0, 10.0, 10.0]),
        exp_status=[0], exp_x=None, x_tol=0.01, obj_tol=1e-2)

    # ---- Problem 4: unbounded var (large finite bounds), x=[0,2], obj=2 ----
    add("P4", 2, 1, 0,
        torch.tensor([[1.0, 1.0]]), torch.tensor([2.0]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([2.0, 1.0]), torch.tensor([0.0, -1e8]), torch.tensor([10.0, 1e8]),
        exp_status=[0], exp_x=torch.tensor([0.0, 2.0]), x_tol=0.1, obj_tol=1e-2)

    # ---- Problem 5: tight bounds at optimum, x=[5,5], obj=-10 ----
    add("P5", 2, 1, 0,
        torch.tensor([[1.0, 1.0]]), torch.tensor([1.0]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([-1.0, -1.0]), torch.tensor([0.0, 0.0]), torch.tensor([5.0, 5.0]),
        exp_status=[2], exp_x=torch.tensor([5.0, 5.0]), x_tol=0.01, obj_tol=1e-3)

    # ---- Problem 6: mixed eq+ineq, x=[5,0,0], obj=5 ----
    add("P6", 3, 1, 1,
        torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([2.0]),
        torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([5.0]),
        torch.tensor([1.0, 3.0, 2.0]), torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([10.0, 10.0, 10.0]),
        exp_status=[0], exp_x=torch.tensor([5.0, 0.0, 0.0]), x_tol=0.01, obj_tol=1e-3)

    # ---- Problem 7: PRIMAL INFEASIBLE (x1+x2>=10 AND x1+x2<=5) ----
    add("P7", 2, 2, 0,
        torch.tensor([[1.0, 1.0], [-1.0, -1.0]]), torch.tensor([10.0, -5.0]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]),
        exp_status=[1, 3], exp_x=None)

    # ---- Problem 8: DUAL INFEASIBLE (min -x1, x1>=x2, x>=0, unbounded) ----
    add("P8", 2, 1, 0,
        torch.tensor([[1.0, -1.0]]), torch.tensor([0.0]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 0.0]),
        torch.tensor([float('inf'), float('inf')]),
        exp_status=[2, 3], exp_x=None)

    # ---- Problem 9: trivial n==0, feasible -> optimal, obj=0 ----
    add("P9", 0, 1, 0,
        torch.tensor([]).reshape(1, 0), torch.tensor([0.0]),
        torch.tensor([]).reshape(0, 0), torch.tensor([]),
        torch.tensor([]), torch.tensor([]), torch.tensor([]),
        exp_status=[0], exp_x=None, obj_tol=1e-6)

    # ---- Problem 10: trivial n==0, infeasible (0>=1) -> primal_infeasible ----
    add("P10", 0, 1, 0,
        torch.tensor([]).reshape(1, 0), torch.tensor([1.0]),
        torch.tensor([]).reshape(0, 0), torch.tensor([]),
        torch.tensor([]), torch.tensor([]), torch.tensor([]),
        exp_status=[1], exp_x=None)

    # ---- Problem 11: trivial m==0, bounded -> optimal, x=[0,0], obj=0 ----
    add("P11", 2, 0, 0,
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([2.0, 1.0]), torch.tensor([0.0, 0.0]), torch.tensor([10.0, 10.0]),
        exp_status=[0], exp_x=torch.tensor([0.0, 0.0]), x_tol=0.01, obj_tol=1e-3)

    # ---- Problem 12: trivial m==0, unbounded -> dual_infeasible ----
    add("P12", 2, 0, 0,
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([]).reshape(0, 2), torch.tensor([]),
        torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 0.0]),
        torch.tensor([float('inf'), float('inf')]),
        exp_status=[2], exp_x=None)

    # ---- Problem 13: transportation (10 suppliers x 15 customers), 150 vars ----
    torch.manual_seed(42)
    n_suppliers = 10
    n_customers = 15
    n_vars = n_suppliers * n_customers
    supply = torch.rand(n_suppliers, dtype=torch.float32) * 20 + 10
    demand = torch.rand(n_customers, dtype=torch.float32) * 15 + 5
    total_demand = demand.sum()
    total_supply = supply.sum()
    if total_supply < total_demand:
        supply = supply * (total_demand / total_supply * 1.2)
    costs = torch.zeros(n_suppliers, n_customers)
    for i in range(n_suppliers):
        for j in range(n_customers):
            costs[i, j] = torch.rand(1, dtype=torch.float32).item() * 5 + abs(i - j) * 0.5
    c13 = costs.flatten()
    G_supply = torch.zeros(n_suppliers, n_vars)
    for i in range(n_suppliers):
        for j in range(n_customers):
            idx = i * n_customers + j
            G_supply[i, idx] = -1.0
    h_supply = -supply
    G_demand = torch.zeros(n_customers, n_vars)
    for j in range(n_customers):
        for i in range(n_suppliers):
            idx = i * n_customers + j
            G_demand[j, idx] = 1.0
    h_demand = demand
    G13 = torch.vstack([G_supply, G_demand])
    h13 = torch.cat([h_supply, h_demand])
    A13 = torch.tensor([]).reshape(0, n_vars)
    b13 = torch.tensor([])
    l13 = torch.zeros(n_vars)
    u13 = torch.ones(n_vars) * float('inf')
    add("P13", n_vars, n_suppliers + n_customers, 0,
        G13, h13, A13, b13, c13, l13, u13,
        exp_status=[0], exp_x=None, x_tol=1e-2, obj_tol=1e-2)

    return problems


if __name__ == "__main__":
    probs = build_problems()
    print(f"# problems: {len(probs)}")
    for p in probs:
        print(f"{p['name']}: n={p['n']} m1={p['m1']} m2={p['m2']} "
              f"exp={p['exp_status']} exp_x={'yes' if p['exp_x'] else 'no'}")
