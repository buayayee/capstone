import sys
import random

sys.path.insert(0, 'preprocessing')
sys.path.insert(0, 'api')

from rule_parser import parse_instructions
from check import check_case, print_case_result, save_results_to_csv, save_results_to_excel
from pathlib import Path

case_num = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(1, 96)

rules = parse_instructions(
    'directives/Main_MP-D 401-01-2005A Deferment Disruption and Exemption Policy for NSmen_050620.pdf'
)
result = check_case(Path(f'submissions/case{case_num}'), rules, use_ai=True)
print_case_result(result)

csv_path   = f'output/result_case{case_num}.csv'
excel_path = f'output/result_case{case_num}.xlsx'
save_results_to_csv([result], csv_path)
print(f'Results saved to: {csv_path}')
try:
    save_results_to_excel([result], excel_path)
    print(f'Results saved to: {excel_path}')
except PermissionError:
    print(f'[WARN] Could not save Excel — {excel_path} may be open. Close the file and retry.')
