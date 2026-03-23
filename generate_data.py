"""
Pre-compute all EMTR scenarios as static JSON for GitHub Pages hosting.

Generates data for all combinations of:
  - 51 states (50 + DC)
  - 5 child counts (0-4)
  - 2 Medicaid options (true/false)
  = 510 total scenarios

Each scenario runs a single PolicyEngine simulation using axes,
replicating the exact logic from the Modal backend (modal_deploy.py v2.9).

Usage:
    python generate_data.py              # Run all scenarios
    python generate_data.py --resume     # Skip already-generated files
    python generate_data.py --state CA   # Run one state only (for testing)
"""

import json
import argparse
import time
from pathlib import Path

from policyengine_us import Simulation


# ── Configuration ──────────────────────────────────────────────────────────
MIN_INCOME = 0
MAX_INCOME = 100_000
STEP = 2_000
YEAR = 2024
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


# ── Correction functions (identical to Modal backend) ──────────────────────

def correct_ctc(employment_income: float, n_children: int, pe_ctc: float) -> float:
    """
    Correct CTC to match actual 2024 rules since PolicyEngine models it
    as fully refundable.

    Actual 2024 CTC:
    - Must have at least $2,500 in earned income for refundable portion
    - Refundable portion (ACTC) = 15% of (earnings - $2,500), max $1,700/child
    - Non-refundable portion requires tax liability (approx: income > $15k)
    """
    if employment_income < 2500:
        return 0.0
    refundable_per_child = min(0.15 * (employment_income - 2500), 1700)
    if employment_income > 15000:
        total_per_child = 2000
    else:
        total_per_child = refundable_per_child
    return total_per_child * n_children


def correct_medicaid(pe_parent_medicaid: float, n_people: int) -> float:
    """
    Correct Medicaid using a flat $5,000 per person imputation.
    If parent is ineligible, zero out the entire household.
    """
    if pe_parent_medicaid > 0:
        return 5000.0 * n_people
    else:
        return 0.0


# ── Main simulation ────────────────────────────────────────────────────────

def run_scenario(state_code: str, n_children: int, include_medicaid: bool) -> list[dict]:
    """Run a single PolicyEngine simulation and return the results list."""
    num_points = (MAX_INCOME - MIN_INCOME) // STEP + 1
    n_people = n_children + 1

    # Build situation with employment_income as an axis
    situation = {
        "people": {
            "parent": {"age": {YEAR: 35}},
        },
        "families": {"family": {"members": ["parent"]}},
        "spm_units": {"spm_unit": {"members": ["parent"]}},
        "households": {
            "household": {
                "members": ["parent"],
                "state_code": {YEAR: state_code},
            }
        },
        "axes": [[{
            "name": "employment_income",
            "min": MIN_INCOME,
            "max": MAX_INCOME,
            "count": num_points,
        }]],
    }

    # Add children
    for i in range(n_children):
        child_id = f"child_{i + 1}"
        situation["people"][child_id] = {"age": {YEAR: 5 + i * 2}}
        situation["families"]["family"]["members"].append(child_id)
        situation["spm_units"]["spm_unit"]["members"].append(child_id)
        situation["households"]["household"]["members"].append(child_id)

    # Run simulation
    sim = Simulation(situation=situation)

    employment_income = sim.calculate("employment_income", YEAR)
    snap = sim.calculate("snap", YEAR)
    tanf = sim.calculate("tanf", YEAR)
    wic = sim.calculate("wic", YEAR)
    eitc = sim.calculate("eitc", YEAR)
    state_eitc = sim.calculate("state_eitc", YEAR)
    ctc = sim.calculate("ctc", YEAR)
    medicaid = sim.calculate("medicaid", YEAR)
    income_tax = sim.calculate("income_tax", YEAR)
    state_income_tax = sim.calculate("state_income_tax", YEAR)

    # Build results (same logic as modal_deploy.py)
    results = []
    for i in range(num_points):
        income = float(employment_income[i * n_people])
        snap_val = float(snap[i])
        tanf_val = float(tanf[i])
        wic_val = float(wic[i])
        eitc_val = float(eitc[i])
        state_eitc_val = float(state_eitc[i])
        income_tax_val = float(income_tax[i])
        state_income_tax_val = float(state_income_tax[i])

        pe_ctc_val = float(ctc[i])
        ctc_val = correct_ctc(income, n_children, pe_ctc_val)

        parent_idx = i * n_people
        pe_parent_medicaid = float(medicaid[parent_idx])

        if include_medicaid:
            medicaid_val = correct_medicaid(pe_parent_medicaid, n_people)
        else:
            medicaid_val = 0.0

        net_income_val = (
            income
            + snap_val
            + tanf_val
            + wic_val
            + eitc_val
            + state_eitc_val
            + ctc_val
            + medicaid_val
            - income_tax_val
            - state_income_tax_val
        )

        # EMTR calculation
        if i < num_points - 1:
            income_next = float(employment_income[(i + 1) * n_people])
            snap_next = float(snap[i + 1])
            tanf_next = float(tanf[i + 1])
            wic_next = float(wic[i + 1])
            eitc_next = float(eitc[i + 1])
            state_eitc_next = float(state_eitc[i + 1])
            income_tax_next = float(income_tax[i + 1])
            state_income_tax_next = float(state_income_tax[i + 1])

            pe_ctc_next = float(ctc[i + 1])
            ctc_next = correct_ctc(income_next, n_children, pe_ctc_next)

            parent_idx_next = (i + 1) * n_people
            pe_parent_medicaid_next = float(medicaid[parent_idx_next])

            if include_medicaid:
                medicaid_next = correct_medicaid(pe_parent_medicaid_next, n_people)
            else:
                medicaid_next = 0.0

            net_income_next = (
                income_next
                + snap_next
                + tanf_next
                + wic_next
                + eitc_next
                + state_eitc_next
                + ctc_next
                + medicaid_next
                - income_tax_next
                - state_income_tax_next
            )

            delta_income = income_next - income
            delta_net = net_income_next - net_income_val
            emtr = 1.0 - (delta_net / delta_income) if delta_income != 0 else None
        else:
            emtr = None

        results.append({
            "employment_income": income,
            "household_net_income": round(net_income_val, 2),
            "snap": round(snap_val, 2),
            "tanf": round(tanf_val, 2),
            "wic": round(wic_val, 2),
            "eitc": round(eitc_val, 2),
            "state_eitc": round(state_eitc_val, 2),
            "ctc": round(ctc_val, 2),
            "medicaid": round(medicaid_val, 2),
            "income_tax": round(income_tax_val, 2),
            "state_income_tax": round(state_income_tax_val, 2),
            "emtr": round(emtr, 6) if emtr is not None else None,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-compute EMTR data for static hosting")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist")
    parser.add_argument("--state", type=str, help="Run one state only (e.g., CA)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Write states.json
    states_list = [{"code": code, "name": name} for code, name in STATES]
    (OUTPUT_DIR / "states.json").write_text(json.dumps(states_list))

    # Filter states if requested
    states_to_run = STATES
    if args.state:
        states_to_run = [(c, n) for c, n in STATES if c == args.state.upper()]
        if not states_to_run:
            print(f"Unknown state: {args.state}")
            return

    children_range = range(5)
    medicaid_options = [True, False]

    total = len(states_to_run) * len(children_range) * len(medicaid_options)
    done = 0
    skipped = 0
    t0 = time.time()

    for state_code, state_name in states_to_run:
        for n_children in children_range:
            for include_medicaid in medicaid_options:
                med_str = "true" if include_medicaid else "false"
                filename = f"{state_code}_{n_children}_{med_str}.json"
                filepath = OUTPUT_DIR / filename

                if args.resume and filepath.exists():
                    skipped += 1
                    done += 1
                    continue

                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 and done > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                print(
                    f"[{done + 1}/{total}] {state_name} | {n_children} children | "
                    f"medicaid={med_str} | "
                    f"~{remaining / 60:.0f}m remaining"
                )

                try:
                    results = run_scenario(state_code, n_children, include_medicaid)
                    filepath.write_text(json.dumps(results))
                    done += 1
                except Exception as e:
                    print(f"  ERROR: {e}")
                    done += 1

    elapsed = time.time() - t0
    print(f"\nDone. {total - skipped} computed, {skipped} skipped. Total time: {elapsed / 60:.1f} minutes.")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
