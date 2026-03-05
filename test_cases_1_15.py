"""
test_cases_1_15.py
------------------
Quick test run of cases 1-15 using the check.py pipeline with RapidOCR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent / "api"))

from check import check_case, print_case_result, save_results_to_csv
from rule_parser import parse_instructions

DIRECTIVE = "directives/Main_MP-D 401-01-2005A Deferment Disruption and Exemption Policy for NSmen_050620.pdf"
SUBMISSIONS = Path("submissions")
OUTPUT_CSV  = "output/results_test_1_15.csv"

rules = parse_instructions(DIRECTIVE)

print(f"\n[Test] Running cases 1-15 with RapidOCR...\n")

cases = []
approved = rejected = warnings = 0

for n in range(1, 16):
    case_dir = SUBMISSIONS / f"case{n}"
    if not case_dir.is_dir():
        print(f"[SKIP] {case_dir} not found")
        continue
    case = check_case(case_dir, rules, use_ai=False)
    print_case_result(case)
    cases.append(case)
    if case.overall_verdict == "APPROVE":
        approved += 1
    elif case.overall_verdict == "REJECT":
        rejected += 1
    else:
        warnings += 1

print("\n" + "="*60)
print(f"  TEST SUMMARY: cases 1-15  ({len(cases)} processed)")
print(f"  [APPROVE]  : {approved}")
print(f"  [REJECT]   : {rejected}")
print(f"  [WARNING]  : {warnings}")
print("="*60)

save_results_to_csv(cases, OUTPUT_CSV)
print(f"\nResults saved to: {OUTPUT_CSV}")
