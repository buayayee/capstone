"""Quick smoke test for extract_uen edge cases."""
import sys
sys.path.insert(0, '/app/api')
from acra_checker import extract_uen

tests = [
    ('Co.Reg.No:199103118KGSTReg.No', '199103118K'),
    ('UEN:199103118K',                 '199103118K'),
    ('UEN 199103118 K GST',            '199103118K'),
    ('T22LL1234A',                     'T22LL1234A'),
    ('1234567890K',                    '1234567890K'),
    ('not12345678K9 nope',             None),   # followed by digit → no match
]

all_ok = True
for text, expected in tests:
    got = extract_uen(text)
    ok = got == expected
    all_ok = all_ok and ok
    mark = 'PASS' if ok else 'FAIL'
    print(f'[{mark}]  input={repr(text):<45s}  expected={expected!s:<15s}  got={got}')

sys.exit(0 if all_ok else 1)
