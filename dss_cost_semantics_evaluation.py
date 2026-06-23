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

DEMAND = {
    1: {"Strategic": 60, "Standard": 30, "Spot": 20},
    2: {"Strategic": 50, "Standard": 40, "Spot": 30},
    3: {"Strategic": 40, "Standard": 40, "Spot": 30},
}

INITIAL_INVENTORY = 20
INTERNAL_CAPACITY = 80
EXTERNAL_CAPACITY = 40
MIN_INTERNAL_USE = 60
MAX_INVENTORY = 40

ACTUAL = {
    "internal_prod": 10,
    "external_supply": 18,
    "holding": 1,
    "lost_margin": {"Strategic": 14, "Standard": 9, "Spot": 5},
    "late_admin": {"Strategic": 2, "Standard": 1, "Spot": 0.5},
}

NOTIONAL = {
    "non_delivery": {"Strategic": 120, "Standard": 70, "Spot": 25},
    "late": {"Strategic": 28, "Standard": 14, "Spot": 4},
    "max_inventory_violation": 18,
    "min_capacity_violation": 8,
}

PRODUCTION_OPTIONS = list(range(0, INTERNAL_CAPACITY + STEP, STEP))
EXTERNAL_OPTIONS = list(range(0, EXTERNAL_CAPACITY + STEP, STEP))


@dataclass(frozen=True)
class Weights:
    name: str
    use_actual_production: bool
    use_actual_holding: bool
    non_delivery: dict[str, float]
    late: dict[str, float]
    max_inventory_violation: float
    min_capacity_violation: float


MODELS = [
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
        non_delivery=NOTIONAL["non_delivery"],
        late=NOTIONAL["late"],
        max_inventory_violation=NOTIONAL["max_inventory_violation"],
        min_capacity_violation=NOTIONAL["min_capacity_violation"],
    ),
    Weights(
        name="C_governed_hybrid",
        use_actual_production=True,
        use_actual_holding=True,
        non_delivery=NOTIONAL["non_delivery"],
        late=NOTIONAL["late"],
        max_inventory_violation=NOTIONAL["max_inventory_violation"],
        min_capacity_violation=NOTIONAL["min_capacity_violation"],
    ),
]


def allocate_period(
    period: int,
    inventory: int,
    backlog: dict[str, int],
    prod_internal: int,
    prod_external: int,
    weights: Weights,
) -> tuple[float, dict[str, int], int, dict]:
    available = inventory + prod_internal + prod_external
    delivered_late = {cls: 0 for cls in CLASSES}
    delivered_ontime = {cls: 0 for cls in CLASSES}

    candidates = []
    for cls in CLASSES:
        if backlog[cls]:
            benefit = weights.non_delivery[cls] - weights.late[cls]
            candidates.append((benefit, "late", cls, backlog[cls]))
    for cls in CLASSES:
        if DEMAND[period][cls]:
            if period == PERIODS[-1]:
                benefit = weights.non_delivery[cls]
            else:
                benefit = max(weights.non_delivery[cls] - weights.late[cls], 0)
            candidates.append((benefit, "ontime", cls, DEMAND[period][cls]))

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
    expired_non_delivery = {
        cls: backlog[cls] - delivered_late[cls] for cls in CLASSES
    }
    current_unmet = {
        cls: DEMAND[period][cls] - delivered_ontime[cls] for cls in CLASSES
    }

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
        period_cost += max(0, ending_inventory - MAX_INVENTORY) * weights.max_inventory_violation

    record = {
        "period": period,
        "internal_production": prod_internal,
        "external_supply": prod_external,
        "available_supply": inventory + prod_internal + prod_external,
        "ending_inventory": ending_inventory,
        **{f"ontime_{cls}": delivered_ontime[cls] for cls in CLASSES},
        **{f"late_{cls}": delivered_late[cls] for cls in CLASSES},
        **{f"non_delivery_expired_{cls}": expired_non_delivery[cls] for cls in CLASSES},
        **{f"new_backlog_{cls}": next_backlog[cls] for cls in CLASSES},
        **{f"terminal_non_delivery_{cls}": current_unmet[cls] if period == PERIODS[-1] else 0 for cls in CLASSES},
    }
    return period_cost, next_backlog, ending_inventory, record


def allocation_cost(
    plan: dict[int, tuple[int, int]],
    weights: Weights,
) -> tuple[float, dict]:
    cost = 0.0
    records = []
    inventory = INITIAL_INVENTORY
    backlog = {cls: 0 for cls in CLASSES}
    for period in PERIODS:
        prod_internal, prod_external = plan[period]
        period_cost, backlog, inventory, record = allocate_period(
            period, inventory, backlog, prod_internal, prod_external, weights
        )
        cost += period_cost
        records.append(record)
    return cost, {"records": records, "end_state": (inventory, tuple(backlog[cls] for cls in CLASSES))}


def production_cost(plan: dict[int, tuple[int, int]], weights: Weights) -> float:
    cost = 0.0
    for period, (internal, external) in plan.items():
        if weights.use_actual_production:
            cost += internal * ACTUAL["internal_prod"] + external * ACTUAL["external_supply"]
        if weights.min_capacity_violation:
            cost += max(0, MIN_INTERNAL_USE - internal) * weights.min_capacity_violation
    return cost


def evaluate_components(records: list[dict]) -> dict:
    actual_financial = 0.0
    notional_penalty = 0.0
    max_inventory_violation_units = 0
    min_capacity_shortfall_units = 0

    totals = {
        "on_time_units": 0,
        "late_units": 0,
        "non_delivered_units": 0,
        "strategic_non_delivered": 0,
        "standard_non_delivered": 0,
        "spot_non_delivered": 0,
        "ending_inventory_total": 0,
        "internal_production": 0,
        "external_supply": 0,
    }

    for row in records:
        internal = row["internal_production"]
        external = row["external_supply"]
        actual_financial += internal * ACTUAL["internal_prod"] + external * ACTUAL["external_supply"]
        actual_financial += row["ending_inventory"] * ACTUAL["holding"]
        totals["internal_production"] += internal
        totals["external_supply"] += external
        totals["ending_inventory_total"] += row["ending_inventory"]
        max_inventory_violation_units += max(0, row["ending_inventory"] - MAX_INVENTORY)
        min_capacity_shortfall_units += max(0, MIN_INTERNAL_USE - internal)

        for cls in CLASSES:
            late = row[f"late_{cls}"]
            non = row[f"non_delivery_expired_{cls}"] + row[f"terminal_non_delivery_{cls}"]
            ontime = row[f"ontime_{cls}"]
            totals["on_time_units"] += ontime
            totals["late_units"] += late
            totals["non_delivered_units"] += non
            totals[f"{cls.lower()}_non_delivered"] += non
            actual_financial += late * ACTUAL["late_admin"][cls]
            actual_financial += non * ACTUAL["lost_margin"][cls]
            notional_penalty += late * NOTIONAL["late"][cls]
            notional_penalty += non * NOTIONAL["non_delivery"][cls]

    notional_penalty += max_inventory_violation_units * NOTIONAL["max_inventory_violation"]
    notional_penalty += min_capacity_shortfall_units * NOTIONAL["min_capacity_violation"]

    return {
        **totals,
        "actual_financial_cost": actual_financial,
        "notional_penalty_score": notional_penalty,
        "max_inventory_violation_units": max_inventory_violation_units,
        "min_capacity_shortfall_units": min_capacity_shortfall_units,
    }


def solve_model(weights: Weights) -> tuple[dict, list[dict]]:
    best = None
    for values in product(PRODUCTION_OPTIONS, EXTERNAL_OPTIONS, repeat=len(PERIODS)):
        plan = {
            period: (values[(period - 1) * 2], values[(period - 1) * 2 + 1])
            for period in PERIODS
        }
        allocation_obj, allocation = allocation_cost(plan, weights)
        obj = production_cost(plan, weights) + allocation_obj
        if best is None or obj < best["objective_value"]:
            components = evaluate_components(allocation["records"])
            best = {
                "model": weights.name,
                "objective_value": obj,
                **components,
            }
            best_records = allocation["records"]
    return best, best_records


def main() -> None:
    summaries = []
    detailed = []
    for weights in MODELS:
        summary, records = solve_model(weights)
        summaries.append(summary)
        for record in records:
            detailed.append({"model": weights.name, **record})

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(detailed)

    summary_df.to_csv(OUT / "DSS_Cost_Semantics_Evaluation_Summary.csv", index=False)
    detail_df.to_csv(OUT / "DSS_Cost_Semantics_Evaluation_Details.csv", index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
