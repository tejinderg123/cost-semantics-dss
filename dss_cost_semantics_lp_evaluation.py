from __future__ import annotations

# Reproducibility package refresh: 2026-06-22 22:57 -04:00.

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import pulp
except ModuleNotFoundError:
    solver_paths = [
        Path("work") / "solver_deps",
        Path(__file__).resolve().parents[2] / "work" / "solver_deps",
    ]
    for solver_path in solver_paths:
        if solver_path.exists():
            sys.path.insert(0, str(solver_path))
            break
    import pulp  # noqa: E402


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ModelDesign:
    name: str
    include_transactional: bool
    include_notional: bool
    service_uses_actual_loss: bool = False


DESIGNS = [
    ModelDesign("A_transactional_only", include_transactional=True, include_notional=False, service_uses_actual_loss=True),
    ModelDesign("B_notional_only", include_transactional=False, include_notional=True),
    ModelDesign("C_governed_hybrid", include_transactional=True, include_notional=True),
]


def solve_supply_lp(
    scenario_name: str,
    products: list[str],
    locations: list[str],
    periods: list[int],
    demand_classes: list[str],
    demand: dict[tuple[str, str, int, str], float],
    internal_capacity: dict[tuple[str, int], float],
    external_capacity: dict[tuple[str, int], float],
    initial_inventory: dict[tuple[str, str], float],
    max_inventory: dict[tuple[str, str, int], float],
    min_internal_use: dict[tuple[str, int], float],
    actual: dict,
    notional: dict,
    design: ModelDesign,
) -> tuple[dict, pd.DataFrame]:
    safe_scenario = "".join(ch if ch.isalnum() else "_" for ch in scenario_name).strip("_")
    prob = pulp.LpProblem(f"{safe_scenario}_{design.name}", pulp.LpMinimize)

    internal = pulp.LpVariable.dicts("internal", (products, locations, periods), lowBound=0)
    external = pulp.LpVariable.dicts("external", (products, locations, periods), lowBound=0)
    inventory = pulp.LpVariable.dicts("inventory", (products, locations, periods), lowBound=0)
    ontime = pulp.LpVariable.dicts("ontime", (products, locations, periods, demand_classes), lowBound=0)
    unserved = pulp.LpVariable.dicts("unserved", (products, locations, periods, demand_classes), lowBound=0)
    max_inv_violation = pulp.LpVariable.dicts("max_inv_violation", (products, locations, periods), lowBound=0)
    min_cap_violation = pulp.LpVariable.dicts("min_cap_violation", (locations, periods), lowBound=0)

    objective_terms = []
    if design.include_transactional:
        objective_terms.extend(
            actual["internal_prod"][(p, l)] * internal[p][l][t]
            + actual["external_supply"][(p, l)] * external[p][l][t]
            + actual["holding"][(p, l)] * inventory[p][l][t]
            for p in products
            for l in locations
            for t in periods
        )
    if design.service_uses_actual_loss:
        objective_terms.extend(
            actual["lost_margin"][k] * unserved[p][l][t][k]
            for p in products
            for l in locations
            for t in periods
            for k in demand_classes
        )
    if design.include_notional:
        objective_terms.extend(
            notional["non_delivery"][k] * unserved[p][l][t][k]
            for p in products
            for l in locations
            for t in periods
            for k in demand_classes
        )
        objective_terms.extend(
            notional["max_inventory_violation"] * max_inv_violation[p][l][t]
            for p in products
            for l in locations
            for t in periods
        )
        objective_terms.extend(
            notional["min_capacity_violation"] * min_cap_violation[l][t]
            for l in locations
            for t in periods
        )
    prob += pulp.lpSum(objective_terms)

    for p in products:
        for l in locations:
            for t in periods:
                prob += internal[p][l][t] <= internal_capacity[(l, t)]
                prob += external[p][l][t] <= external_capacity[(l, t)]
                inbound = initial_inventory[(p, l)] if t == periods[0] else inventory[p][l][t - 1]
                prob += (
                    inbound + internal[p][l][t] + external[p][l][t]
                    == pulp.lpSum(ontime[p][l][t][k] for k in demand_classes) + inventory[p][l][t]
                )
                prob += inventory[p][l][t] - max_inv_violation[p][l][t] <= max_inventory[(p, l, t)]
                for k in demand_classes:
                    prob += ontime[p][l][t][k] + unserved[p][l][t][k] == demand[(p, l, t, k)]

    for l in locations:
        for t in periods:
            prob += (
                pulp.lpSum(internal[p][l][t] for p in products) + min_cap_violation[l][t]
                >= min_internal_use[(l, t)]
            )

    start = time.perf_counter()
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_seconds = time.perf_counter() - start
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        raise RuntimeError(f"{scenario_name} {design.name} not optimal: {status_name}")

    rows = []
    for p in products:
        for l in locations:
            for t in periods:
                row = {
                    "scenario": scenario_name,
                    "model": design.name,
                    "product": p,
                    "location": l,
                    "period": t,
                    "internal": pulp.value(internal[p][l][t]),
                    "external": pulp.value(external[p][l][t]),
                    "inventory": pulp.value(inventory[p][l][t]),
                    "max_inventory_violation": pulp.value(max_inv_violation[p][l][t]),
                }
                for k in demand_classes:
                    row[f"ontime_{k}"] = pulp.value(ontime[p][l][t][k])
                    row[f"unserved_{k}"] = pulp.value(unserved[p][l][t][k])
                rows.append(row)
    detail = pd.DataFrame(rows)

    actual_financial = 0.0
    notional_penalty = 0.0
    on_time_units = 0.0
    unserved_units = 0.0
    for _, row in detail.iterrows():
        p, l, t = row["product"], row["location"], row["period"]
        actual_financial += actual["internal_prod"][(p, l)] * row["internal"]
        actual_financial += actual["external_supply"][(p, l)] * row["external"]
        actual_financial += actual["holding"][(p, l)] * row["inventory"]
        notional_penalty += notional["max_inventory_violation"] * row["max_inventory_violation"]
        for k in demand_classes:
            on_time_units += row[f"ontime_{k}"]
            unserved_units += row[f"unserved_{k}"]
            actual_financial += actual["lost_margin"][k] * row[f"unserved_{k}"]
            notional_penalty += notional["non_delivery"][k] * row[f"unserved_{k}"]
    min_shortfall = 0.0
    for l in locations:
        for t in periods:
            used = sum(detail[(detail["location"] == l) & (detail["period"] == t)]["internal"])
            shortfall = max(0.0, min_internal_use[(l, t)] - used)
            min_shortfall += shortfall
            notional_penalty += notional["min_capacity_violation"] * shortfall

    summary = {
        "scenario": scenario_name,
        "model": design.name,
        "status": status_name,
        "objective_value": pulp.value(prob.objective),
        "on_time_units": on_time_units,
        "unserved_units": unserved_units,
        "internal_production": detail["internal"].sum(),
        "external_supply": detail["external"].sum(),
        "actual_financial_cost": actual_financial,
        "notional_penalty_score": notional_penalty,
        "max_inventory_violation_units": detail["max_inventory_violation"].sum(),
        "min_capacity_shortfall_units": min_shortfall,
        "variables": len(prob.variables()),
        "constraints": len(prob.constraints),
        "solve_seconds": solve_seconds,
    }
    return summary, detail


def baseline_data(non_delivery_factor=1.0, external_cap=40, demand_factor=1.0):
    products = ["P1"]
    locations = ["L1"]
    periods = [1, 2, 3]
    classes = ["Strategic", "Standard", "Spot"]
    base_demand = {
        1: {"Strategic": 60, "Standard": 30, "Spot": 20},
        2: {"Strategic": 50, "Standard": 40, "Spot": 30},
        3: {"Strategic": 40, "Standard": 40, "Spot": 30},
    }
    demand = {
        ("P1", "L1", t, k): round(qty * demand_factor, 2)
        for t, row in base_demand.items()
        for k, qty in row.items()
    }
    actual = {
        "internal_prod": {("P1", "L1"): 10},
        "external_supply": {("P1", "L1"): 18},
        "holding": {("P1", "L1"): 1},
        "lost_margin": {"Strategic": 14, "Standard": 9, "Spot": 5},
    }
    notional = {
        "non_delivery": {
            "Strategic": 120 * non_delivery_factor,
            "Standard": 70 * non_delivery_factor,
            "Spot": 25 * non_delivery_factor,
        },
        "max_inventory_violation": 18,
        "min_capacity_violation": 8,
    }
    internal_capacity = {("L1", t): 80 for t in periods}
    external_capacity = {("L1", t): external_cap for t in periods}
    initial_inventory = {("P1", "L1"): 20}
    max_inventory = {("P1", "L1", t): 40 for t in periods}
    min_internal_use = {("L1", t): 60 for t in periods}
    return products, locations, periods, classes, demand, internal_capacity, external_capacity, initial_inventory, max_inventory, min_internal_use, actual, notional


def scaled_data(seed=7):
    random.seed(seed)
    products = [f"P{i:02d}" for i in range(1, 21)]
    locations = [f"L{i:02d}" for i in range(1, 11)]
    periods = list(range(1, 13))
    classes = ["Strategic", "Standard", "Spot"]
    demand = {}
    actual = {"internal_prod": {}, "external_supply": {}, "holding": {}, "lost_margin": {"Strategic": 14, "Standard": 9, "Spot": 5}}
    notional = {
        "non_delivery": {"Strategic": 120, "Standard": 70, "Spot": 25},
        "max_inventory_violation": 18,
        "min_capacity_violation": 8,
    }
    internal_capacity = {}
    external_capacity = {}
    initial_inventory = {}
    max_inventory = {}
    min_internal_use = {}
    for l in locations:
        for t in periods:
            internal_capacity[(l, t)] = random.randint(600, 900)
            external_capacity[(l, t)] = random.randint(250, 450)
            min_internal_use[(l, t)] = int(internal_capacity[(l, t)] * 0.45)
    for p in products:
        for l in locations:
            actual["internal_prod"][(p, l)] = random.randint(8, 13)
            actual["external_supply"][(p, l)] = random.randint(15, 23)
            actual["holding"][(p, l)] = round(random.uniform(0.6, 1.5), 2)
            initial_inventory[(p, l)] = random.randint(5, 30)
            for t in periods:
                max_inventory[(p, l, t)] = random.randint(60, 120)
                for k in classes:
                    base = {"Strategic": 20, "Standard": 15, "Spot": 10}[k]
                    demand[(p, l, t, k)] = random.randint(max(1, base - 6), base + 10)
    return products, locations, periods, classes, demand, internal_capacity, external_capacity, initial_inventory, max_inventory, min_internal_use, actual, notional


def run_case(name, data):
    summaries = []
    for design in DESIGNS:
        summary, _ = solve_supply_lp(name, *data, design)
        summaries.append(summary)
    return summaries


def main():
    all_summaries = []

    sensitivity = [
        ("Baseline", baseline_data()),
        ("Non-delivery penalty 50%", baseline_data(non_delivery_factor=0.5)),
        ("Non-delivery penalty 200%", baseline_data(non_delivery_factor=2.0)),
        ("External capacity tight", baseline_data(external_cap=20)),
        ("External capacity expanded", baseline_data(external_cap=60)),
        ("Demand -10%", baseline_data(demand_factor=0.9)),
        ("Demand +10%", baseline_data(demand_factor=1.1)),
    ]
    for name, data in sensitivity:
        summaries = run_case(name, data)
        all_summaries.extend(summaries)

    scaled_summaries = run_case("Scaled stress test", scaled_data())
    all_summaries.extend(scaled_summaries)

    summary_df = pd.DataFrame(all_summaries)
    baseline = summary_df[summary_df["scenario"] == "Baseline"]
    sensitivity_c = summary_df[(summary_df["scenario"] != "Scaled stress test") & (summary_df["model"] == "C_governed_hybrid")]
    scaled = summary_df[summary_df["scenario"] == "Scaled stress test"]
    baseline.to_csv(OUT / "DSS_Cost_Semantics_Evaluation_Summary.csv", index=False)
    sensitivity_c.to_csv(OUT / "DSS_Cost_Semantics_Sensitivity_Results.csv", index=False)
    scaled.to_csv(OUT / "DSS_Cost_Semantics_Scaled_Stress_Test.csv", index=False)

    print("BASELINE")
    print(baseline.to_string(index=False))
    print("\nSENSITIVITY MODEL C")
    print(sensitivity_c.to_string(index=False))
    print("\nSCALED")
    print(scaled[["model", "status", "variables", "constraints", "solve_seconds", "actual_financial_cost", "notional_penalty_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
