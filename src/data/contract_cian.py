"""Data Contract checks for fresh CIAN real estate listings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ContractViolation:
    field: str
    rule: str
    details: str
    severity: str = "error"


CIAN_DATA_CONTRACT = {
    "listing_id": {"required": True, "max_null_pct": 0.01},
    "url": {"required": True, "max_null_pct": 0.0},
    "source": {"required": True, "allowed_values": ["cian"], "max_null_pct": 0.0},
    "collected_at": {"required": True, "max_null_pct": 0.0},
    "location_query": {"required": True, "allowed_values": ["Санкт-Петербург"], "max_null_pct": 0.0},
    "price": {"required": True, "min": 1_000_000, "max": 600_000_000, "max_null_pct": 0.0},
    "total_meters": {"required": True, "min": 10, "max": 500, "max_null_pct": 0.0},
    "room_segment": {
        "required": False,
        "allowed_values": ["studio", "1room", "2rooms", "3rooms", "4rooms", "unknown"],
        "max_null_pct": 0.05,
    },
    "rooms_count": {"required": True, "min": 0, "max": 10, "max_null_pct": 0.02},
    "floor": {"required": False, "min": 1, "max": 100, "max_null_pct": 0.25},
    "floors_count": {"required": False, "min": 1, "max": 100, "max_null_pct": 0.25},
    # district enforces VALID_SPB_DISTRICTS via clean_cian.py; observed 0% null after broken-row filter.
    "district": {"required": True, "max_null_pct": 0.01},
    # underground is always present as a literal string; ~3.6% rows carry the placeholder "unknown".
    "underground": {"required": True, "max_null_pct": 0.05},
    "residential_complex": {"required": False, "max_null_pct": 0.70},
    # Geocoding-derived features (added in geocoded dataset; optional in contract for raw clean.csv).
    "lat": {"required": False, "min": 59.5, "max": 60.3, "max_null_pct": 0.0},
    "lon": {"required": False, "min": 29.4, "max": 30.9, "max_null_pct": 0.0},
    "geo_precision": {
        "required": False,
        "allowed_values": ["house", "street", "district"],
        "max_null_pct": 0.0,
    },
    "distance_to_center_km": {"required": False, "min": 0.0, "max": 50.0, "max_null_pct": 0.0},
    # NaN allowed when the listing did not name a metro station (unknown ≈ 3.6%).
    # Upper bound is 50 km to accommodate suburban districts (Курортный, Кронштадтский,
    # Петродворцовый) where the named metro station is far from the listing.
    "distance_to_metro_km": {"required": False, "min": 0.0, "max": 50.0, "max_null_pct": 0.05},
    # Real pedestrian routing is optional because it requires an external API key.
    "distance_to_metro_route_km": {"required": False, "min": 0.0, "max": 70.0, "max_null_pct": 1.0},
    "duration_to_metro_route_min": {"required": False, "min": 0.0, "max": 720.0, "max_null_pct": 1.0},
    "metro_known": {"required": False, "max_null_pct": 0.0},
    # Active modeling target derived in build_features.py; optional in raw clean.csv.
    "target_price_per_sqm": {"required": False, "min": 50_000, "max": 3_000_000, "max_null_pct": 0.0},
}


def validate_contract(df: pd.DataFrame) -> list[ContractViolation]:
    """Validate a DataFrame against the CIAN contract."""
    violations: list[ContractViolation] = []

    for field, rules in CIAN_DATA_CONTRACT.items():
        if field not in df.columns:
            if rules.get("required", False):
                violations.append(ContractViolation(field, "required_field", f"Missing required field: {field}"))
            continue

        series = df[field]
        null_pct = series.isna().mean()
        max_null_pct = rules.get("max_null_pct", 1.0)
        if null_pct > max_null_pct:
            severity = "warning" if not rules.get("required", False) else "error"
            violations.append(
                ContractViolation(
                    field,
                    "null_percentage",
                    f"Null share {null_pct:.2%} > allowed {max_null_pct:.2%}",
                    severity=severity,
                )
            )

        non_null = series.dropna()
        if "min" in rules and len(non_null) > 0:
            numeric = pd.to_numeric(non_null, errors="coerce")
            count = numeric.lt(rules["min"]).sum()
            if count:
                violations.append(ContractViolation(field, "range_min", f"{count} values below {rules['min']}"))

        if "max" in rules and len(non_null) > 0:
            numeric = pd.to_numeric(non_null, errors="coerce")
            count = numeric.gt(rules["max"]).sum()
            if count:
                violations.append(ContractViolation(field, "range_max", f"{count} values above {rules['max']}"))

        if "allowed_values" in rules and len(non_null) > 0:
            invalid = ~non_null.isin(rules["allowed_values"])
            if invalid.sum():
                violations.append(
                    ContractViolation(
                        field,
                        "allowed_values",
                        f"{invalid.sum()} values outside {rules['allowed_values']}",
                    )
                )

    return violations


def print_report(violations: list[ContractViolation]) -> None:
    """Print a compact validation report."""
    if not violations:
        print("Data Contract: OK")
        return

    errors = [item for item in violations if item.severity == "error"]
    warnings = [item for item in violations if item.severity == "warning"]
    print(f"Data Contract: {len(errors)} errors, {len(warnings)} warnings")
    for item in violations:
        marker = "ERROR" if item.severity == "error" else "WARN"
        print(f"[{marker}] {item.field}: {item.details} ({item.rule})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CIAN data contract.")
    parser.add_argument("path", nargs="?", default="data/processed/cian_spb_clean.csv")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.path))
    print_report(validate_contract(df))


if __name__ == "__main__":
    main()
