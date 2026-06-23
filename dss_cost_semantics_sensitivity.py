from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

PERIODS = [1, 2, 3]
CLASSES = ["Strategic", "Standard", "Spot"]
STEP = 10

BASE_DEMAND = {
    1: {"Strategic": 60, "Standard": 30, "Spot": 20},
    2: {"Strategic": 50, "Standard": 40, "Spot": 30},
    3: {"Strategic": 40, "Standard": 40, "Spot": 30},
}

ACTUAL = {
    "internal_prod": 10,
    "external_supply": 18,
    "holding": 1,
    "lost_margin": {"Strategic": 14, "Standard": 9, "Spot": 5},
    "late_admin": {"Strategic": 2, "Standard": 1, "Spot": 0.5},
}

BASE_NOTIONAL = {
    "non_delivery": {"Strategic": 120, "Standard": 70, "Spot": 25},
    "late": {"Strategic": 28, "Standard": 14, "Spot": 4},
    "max_inventory_violation": 18,
    "min_capacity_violation": 8,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    non_delivery_factor: float = 1.0
    external_capacity: int = 40
    demand_factor: float = 1.0


@dataclass(frozen=True)
class Weights:
    name: str
    use_actual_production: bool
    use_actual_holding: bool
    non_delivery: dict[str, float]
    late: dict[str, float]
    max_inventory_violation: float
    min_capacity_violation: float


def scaled_demand(factor: float) -> dict[int, dict[str, int]]:
    demand: dict[int, dict[str, int]] = {}
    for period, row in BASE_DEMAND.items():
        demand[period] = {
            cls: int(round((qty * factor) / STEP) * STEP)
            for cls, qty in row.items()
        }
    return demand


def build_models(scenario: Scenario) -> list[Weights]:
    notional = {
        "non_delivery": {
            cls: cost * scenario.non_delivery_factor
            for cls, cost in BASE_NOTIONAL["non_delivery"].items()
        },
        "late": BASE_NOTIONAL["late"],
        "max_inventory_violation": BASE_NOTIONAL["max_inventory_violation"],
        "min_capacity_violation": BASE_NOTIONAL["min_capacity_violation"],
    }
    return [
        Weights(
            name="A_transactional_only",
            use_actual_production=True,
            use_actual_holding=True,
            non_delivery=ACTUAL["lost_margin"],
            late=ACTUAL["late_admin"],
            max_inventory_violation=0,
            min_capacity_violation=0,
        ),
        Weights(
            name="B_notional_only",
            use_actual_production=False,
            use_actual_holding=False,
            non_delivery=notional["non_delivery"],
            late=notional["late"],
            max_inventory_violation=notional["max_inventory_violation"],
            min_capacity_violation=notional["min_capacity_violation"],
        ),
        Weights(
            name="C_governed_hybrid",
            use_actual_production=True,
            use_actual_holding=True,
            non_delivery=notional["non_delivery"],
            late=notional["late"],
            max_inventory_violation=notional["max_inventory_violation"],
            min_capacity_violation=notional["min_capacity_violation"],
        ),
    ]


def allocate_period(period, inventory, backlog, prod_internal, prod_external, weights, demand):
    available = inventory + prod_internal + prod_external
    delivered_late = {cls: 0 for cls in CLASSES}
    delivered_ontime = {cls: 0 for cls in CLASSES}
    candidates = []
    for cls in CLASSES:
        if backlog[cls]:
            benefit = weights.non_delivery[cls] - weights.late[cls]
            candidates.append((benefit, "late", cls, backlog[cls]))
    for cls in CLASSES:
        if demand[period][cls]:
            benefit = weights.non_delivery[cls] if period == PERIODS[-1] else max(weights.non_delivery[cls] - weights.late[cls], 0)
            candidates.append((benefit, "ontime", cls, demand[period][cls]))

    for _benefit, kind, cls, qty in sorted(candidates, reverse=True):
        deliver = min(qty, available)
        deliver = (deliver // STEP) * STEP
        if deliver <= 0:
            continue
        available -= deliver
        if kind == "late":
            delivered_late[cls] = deliver
        else:
            delivered_ontime[cls] = deliver

    ending_inventory = available
    expired_non_delivery = {cls: backlog[cls] - delivered_late[cls] for cls in CLASSES}
    current_unmet = {cls: demand[period][cls] - delivered_ontime[cls] for cls in CLASSES}

    period_cost = 0.0
    for cls in CLASSES:
        period_cost += delivered_late[cls] * weights.late[cls]
        period_cost += expired_non_delivery[cls] * weights.non_delivery[cls]
    if period == PERIODS[-1]:
        for cls in CLASSES:
            period_cost += current_unmet[cls] * weights.non_delivery[cls]
        next_backlog = {cls: 0 for cls in CLASSES}
    else:
        next_backlog = current_unmet

    if weights.use_actual_holding:
        period_cost += ending_inventory * ACTUAL["holding"]
    if weights.max_inventory_violation:
        period_cost += max(0, ending_inventory - 40) * weights.max_inventory_violation

    record = {
        "period": period,
        "internal_production": prod_internal,
        "external_supply": prod_external,
        "ending_inventory": ending_inventory,
        **{f"ontime_{cls}": delivered_ontime[cls] for cls in CLASSES},
        **{f"late_{cls}": delivered_late[cls] for cls in CLASSES},
        **{f"non_delivery_expired_{cls}": expired_non_delivery[cls] for cls in CLASSES},
        **{f"terminal_non_delivery_{cls}": current_unmet[cls] if period == PERIODS[-1] else 0 for cls in CLASSES},
    }
    return period_cost, next_backlog, ending_inventory, record


def allocation_cost(plan, weights, demand):
    cost = 0.0
    records = []
    inventory = 20
    backlog = {cls: 0 for cls in CLASSES}
    for period in PERIODS:
        internal, external = plan[period]
        period_cost, backlog, inventory, record = allocate_period(period, inventory, backlog, internal, external, weights, demand)
        cost += period_cost
        records.append(record)
    return cost, records


def production_cost(plan, weights):
    cost = 0.0
    for _period, (internal, external) in plan.items():
        if weights.use_actual_production:
            cost += internal * ACTUAL["internal_prod"] + external * ACTUAL["external_supply"]
        if weights.min_capacity_violation:
            cost += max(0, 60 - internal) * weights.min_capacity_violation
    return cost


def evaluate_components(records):
    totals = {
        "on_time_units": 0,
        "late_units": 0,
        "non_delivered_units": 0,
        "internal_production": 0,
        "external_supply": 0,
    }
    actual_financial = 0.0
    notional_penalty = 0.0
    min_capacity_shortfall = 0
    max_inventory_violation = 0
    for row in records:
        actual_financial += row["internal_production"] * ACTUAL["internal_prod"]
        actual_financial += row["external_supply"] * ACTUAL["external_supply"]
        actual_financial += row["ending_inventory"] * ACTUAL["holding"]
        totals["internal_production"] += row["internal_production"]
        totals["external_supply"] += row["external_supply"]
        min_capacity_shortfall += max(0, 60 - row["internal_production"])
        max_inventory_violation += max(0, row["ending_inventory"] - 40)
        for cls in CLASSES:
            late = row[f"late_{cls}"]
            non = row[f"non_delivery_expired_{cls}"] + row[f"terminal_non_delivery_{cls}"]
            ontime = row[f"ontime_{cls}"]
            totals["on_time_units"] += ontime
            totals["late_units"] += late
            totals["non_delivered_units"] += non
            actual_financial += late * ACTUAL["late_admin"][cls] + non * ACTUAL["lost_margin"][cls]
            notional_penalty += late * BASE_NOTIONAL["late"][cls] + non * BASE_NOTIONAL["non_delivery"][cls]
    notional_penalty += min_capacity_shortfall * BASE_NOTIONAL["min_capacity_violation"]
    notional_penalty += max_inventory_violation * BASE_NOTIONAL["max_inventory_violation"]
    return {**totals, "actual_financial_cost": actual_financial, "notional_penalty_score": notional_penalty}


def solve(scenario: Scenario, weights: Weights):
    demand = scaled_demand(scenario.demand_factor)
    internal_options = list(range(0, 80 + STEP, STEP))
    external_options = list(range(0, scenario.external_capacity + STEP, STEP))
    best = None
    for values in product(internal_options, external_options, repeat=len(PERIODS)):
        plan = {period: (values[(period - 1) * 2], values[(period - 1) * 2 + 1]) for period in PERIODS}
        allocation_obj, records = allocation_cost(plan, weights, demand)
        obj = production_cost(plan, weights) + allocation_obj
        if best is None or obj < best["objective_value"]:
            best = {
                "scenario": scenario.name,
                "model": weights.name,
                "objective_value": obj,
                **evaluate_components(records),
            }
    return best


def main():
    scenarios = [
        Scenario("Baseline"),
        Scenario("Non-delivery penalty 50%", non_delivery_factor=0.5),
        Scenario("Non-delivery penalty 200%", non_delivery_factor=2.0),
        Scenario("External capacity tight", external_capacity=20),
        Scenario("External capacity expanded", external_capacity=60),
        Scenario("Demand -10%", demand_factor=0.9),
        Scenario("Demand +10%", demand_factor=1.1),
    ]
    rows = []
    for scenario in scenarios:
        for model in build_models(scenario):
            rows.append(solve(scenario, model))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "DSS_Cost_Semantics_Sensitivity_Results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
