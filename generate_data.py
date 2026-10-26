"""
Pre-compute EMTR scenarios as static JSON for GitHub Pages hosting.

Generates one file per combination of
  - tax year (default: 2023, 2024, 2025)
  - 50 states
  - child count 0-4
  - Medicaid included / excluded
= 1,500 scenario files, laid out as data/<year>/<STATE>_<children>_<true|false>.json,
plus data/years.json, data/states.json and data/<year>/meta.json.

Household: one parent aged 35, children aged 5, 7, 9, 11 (so WIC is always zero:
no child under 5 and no pregnancy). Earnings run $0-$100,000 in $2,000 steps.

Net income accounting (changed 2026-10-25):
  net = earnings + SNAP + TANF + WIC + Medicaid value
        - federal income tax - state income tax
  PolicyEngine's `income_tax` is tax before refundable credits MINUS refundable
  credits (EITC, refundable CTC, ...), and `state_income_tax` likewise nets the
  state EITC and other state refundable credits. The earlier version of this
  script added EITC, state EITC and CTC on top of those, double counting them.
  EITC, state EITC and CTC are now exported for display only.

CTC shown = refundable CTC + non-refundable CTC actually used against liability.
(PolicyEngine's `ctc` variable is the post-phase-out maximum before the liability
cap, which is why the old hand "correction" existed.) The script asserts that the
CTC is the only non-refundable credit the household receives, so the realised
non-refundable part equals `income_tax_capped_non_refundable_credits`.

Medicaid value: flat $5,000 per household member when the parent is Medicaid
eligible, $0 otherwise (unchanged from the original design).
Payroll taxes are NOT included in the EMTR (unchanged from the original design).

Usage:
    python generate_data.py                      # all years, all states
    python generate_data.py --years 2025         # one year
    python generate_data.py --state CA           # one state (all years)
    python generate_data.py --resume             # skip files that already exist
Any simulation error aborts the run (no silent gaps).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
from policyengine_us import Simulation

# ── Configuration ──────────────────────────────────────────────────────────
MIN_INCOME = 0
MAX_INCOME = 100_000
STEP = 2_000
DEFAULT_YEARS = [2023, 2024, 2025]
MEDICAID_VALUE_PER_PERSON = 5_000.0
CHILD_AGES = [5, 7, 9, 11]
OUTPUT_DIR = Path(__file__).parent / "data"

STATES = [
    ("AL", "Alabama"), ("AK", "Alaska"), ("AZ", "Arizona"), ("AR", "Arkansas"),
    ("CA", "California"), ("CO", "Colorado"), ("CT", "Connecticut"), ("DE", "Delaware"),
    ("FL", "Florida"), ("GA", "Georgia"), ("HI", "Hawaii"), ("ID", "Idaho"),
    ("IL", "Illinois"), ("IN", "Indiana"), ("IA", "Iowa"), ("KS", "Kansas"),
    ("KY", "Kentucky"), ("LA", "Louisiana"), ("ME", "Maine"), ("MD", "Maryland"),
    ("MA", "Massachusetts"), ("MI", "Michigan"), ("MN", "Minnesota"), ("MS", "Mississippi"),
    ("MO", "Missouri"), ("MT", "Montana"), ("NE", "Nebraska"), ("NV", "Nevada"),
    ("NH", "New Hampshire"), ("NJ", "New Jersey"), ("NM", "New Mexico"), ("NY", "New York"),
    ("NC", "North Carolina"), ("ND", "North Dakota"), ("OH", "Ohio"), ("OK", "Oklahoma"),
    ("OR", "Oregon"), ("PA", "Pennsylvania"), ("RI", "Rhode Island"), ("SC", "South Carolina"),
    ("SD", "South Dakota"), ("TN", "Tennessee"), ("TX", "Texas"), ("UT", "Utah"),
    ("VT", "Vermont"), ("VA", "Virginia"), ("WA", "Washington"), ("WV", "West Virginia"),
    ("WI", "Wisconsin"), ("WY", "Wyoming"),
]

# Unit-level variables pulled from PolicyEngine (WIC is person-level and is summed per household)
UNIT_VARS = [
    "snap", "tanf", "wic", "eitc", "state_eitc",
    "refundable_ctc", "non_refundable_ctc",
    "income_tax_non_refundable_credits", "income_tax_capped_non_refundable_credits",
    "income_tax", "state_income_tax",
]


def build_situation(year: int, state_code: str, n_children: int, num_points: int) -> dict:
    members = ["parent"]
    people = {"parent": {"age": {year: 35}}}
    for i in range(n_children):
        cid = f"child_{i + 1}"
        people[cid] = {"age": {year: CHILD_AGES[i]}}
        members.append(cid)
    return {
        "people": people,
        "families": {"family": {"members": members}},
        "marital_units": {"marital_unit": {"members": ["parent"]}},
        "tax_units": {"tax_unit": {"members": members}},
        "spm_units": {"spm_unit": {"members": members}},
        "households": {
            "household": {"members": members, "state_code": {year: state_code}}
        },
        "axes": [[{
            "name": "employment_income",
            "period": year,  # without this the axis applies to PolicyEngine's default year only
            "min": MIN_INCOME,
            "max": MAX_INCOME,
            "count": num_points,
        }]],
    }


def run_scenario(year: int, state_code: str, n_children: int, include_medicaid: bool) -> list[dict]:
    """Run one PolicyEngine simulation (earnings as an axis) and return the rows."""
    num_points = (MAX_INCOME - MIN_INCOME) // STEP + 1
    n_people = n_children + 1

    sim = Simulation(situation=build_situation(year, state_code, n_children, num_points))

    # Person-level arrays have one row per person per axis point; parent is first.
    parent_rows = np.arange(num_points) * n_people
    income = np.asarray(sim.calculate("employment_income", year), dtype=float)[parent_rows]
    expected_grid = np.linspace(MIN_INCOME, MAX_INCOME, num_points)
    if not np.allclose(income, expected_grid):
        raise RuntimeError(f"{state_code} {year}: earnings axis did not apply (got {income[:3]}...)")
    parent_medicaid = np.asarray(sim.calculate("medicaid", year), dtype=float)[parent_rows]

    u = {}
    for v in UNIT_VARS:
        arr = np.asarray(sim.calculate(v, year), dtype=float)
        if arr.shape == (num_points * n_people,):
            # person-level variable (WIC): sum over household members at each axis point
            arr = arr.reshape(num_points, n_people).sum(axis=1)
        if arr.shape != (num_points,):
            raise RuntimeError(f"{state_code} {year}: {v} has shape {arr.shape}, expected ({num_points},)")
        u[v] = arr

    # Identification check: CTC must be the only non-refundable credit, otherwise the
    # realised non-refundable CTC cannot be read off the capped-credits total.
    if not np.allclose(u["income_tax_non_refundable_credits"], u["non_refundable_ctc"], atol=0.5):
        raise RuntimeError(f"{state_code} {year} {n_children} kids: non-refundable credits other than CTC present")

    ctc_refundable = u["refundable_ctc"]
    ctc_nonrefundable_used = u["income_tax_capped_non_refundable_credits"]
    ctc_total = ctc_refundable + ctc_nonrefundable_used

    if include_medicaid:
        medicaid_value = np.where(parent_medicaid > 0, MEDICAID_VALUE_PER_PERSON * n_people, 0.0)
    else:
        medicaid_value = np.zeros(num_points)

    net = (
        income + u["snap"] + u["tanf"] + u["wic"] + medicaid_value
        - u["income_tax"] - u["state_income_tax"]
    )

    d_income = np.diff(income)
    d_net = np.diff(net)
    emtr = np.full(num_points, np.nan)
    emtr[:-1] = 1.0 - d_net / d_income

    rows = []
    for i in range(num_points):
        rows.append({
            "employment_income": float(income[i]),
            "household_net_income": round(float(net[i]), 2),
            "snap": round(float(u["snap"][i]), 2),
            "tanf": round(float(u["tanf"][i]), 2),
            "wic": round(float(u["wic"][i]), 2),
            "eitc": round(float(u["eitc"][i]), 2),
            "state_eitc": round(float(u["state_eitc"][i]), 2),
            "ctc": round(float(ctc_total[i]), 2),
            "ctc_refundable": round(float(ctc_refundable[i]), 2),
            "ctc_nonrefundable": round(float(ctc_nonrefundable_used[i]), 2),
            "medicaid": round(float(medicaid_value[i]), 2),
            "income_tax": round(float(u["income_tax"][i]), 2),
            "state_income_tax": round(float(u["state_income_tax"][i]), 2),
            "emtr": None if np.isnan(emtr[i]) else round(float(emtr[i]), 6),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute EMTR data for static hosting")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--state", type=str, help="Run one state only (e.g. CA)")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "states.json").write_text(
        json.dumps([{"code": c, "name": n} for c, n in STATES])
    )
    (OUTPUT_DIR / "years.json").write_text(json.dumps(sorted(set(DEFAULT_YEARS) | set(args.years))))

    states_to_run = STATES
    if args.state:
        states_to_run = [(c, n) for c, n in STATES if c == args.state.upper()]
        if not states_to_run:
            print(f"Unknown state: {args.state}")
            return 2

    pe_version = pkg_version("policyengine-us")
    total = len(args.years) * len(states_to_run) * 5 * 2
    done = skipped = 0
    t0 = time.time()

    for year in args.years:
        year_dir = OUTPUT_DIR / str(year)
        year_dir.mkdir(exist_ok=True)
        for state_code, state_name in states_to_run:
            for n_children in range(5):
                for include_medicaid in (True, False):
                    med_str = "true" if include_medicaid else "false"
                    filepath = year_dir / f"{state_code}_{n_children}_{med_str}.json"
                    if args.resume and filepath.exists():
                        skipped += 1
                        done += 1
                        continue
                    elapsed = time.time() - t0
                    computed = done - skipped
                    rate = computed / elapsed if computed > 0 else 0
                    remaining = (total - done) / rate / 60 if rate > 0 else float("nan")
                    print(f"[{done + 1}/{total}] {year} {state_name} | {n_children} kids | "
                          f"medicaid={med_str} | ~{remaining:.0f}m left", flush=True)
                    try:
                        rows = run_scenario(year, state_code, n_children, include_medicaid)
                    except Exception as e:  # abort: no silent gaps
                        print(f"\nABORT on {filepath.name}: {type(e).__name__}: {e}", flush=True)
                        return 1
                    filepath.write_text(json.dumps(rows))
                    done += 1

        (year_dir / "meta.json").write_text(json.dumps({
            "tax_year": year,
            "policyengine_us_version": pe_version,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "income_min": MIN_INCOME, "income_max": MAX_INCOME, "income_step": STEP,
            "medicaid_value_per_person": MEDICAID_VALUE_PER_PERSON,
            "child_ages": CHILD_AGES,
            "payroll_tax_included": False,
            "net_income": "earnings + snap + tanf + wic + medicaid - income_tax - state_income_tax "
                          "(taxes are net of refundable credits incl. EITC and CTC)",
        }, indent=2))

    print(f"\nDone. {done - skipped} computed, {skipped} skipped, {(time.time() - t0) / 60:.1f} min. "
          f"policyengine-us {pe_version}. Output: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
