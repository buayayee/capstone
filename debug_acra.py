import sys; sys.path.insert(0, 'api')
from acra_checker import _normalize_name, _query, _name_score
import json

name = 'Tech Data Distribution (Singapore) Pte Ltd'
postal = '486066'

# Strategy A: filter directly by postal code
rows_postal = _query({'filters': json.dumps({'reg_postal_code': postal}), 'limit': 10})
print(f'Direct postal filter ({postal}):')
for r in rows_postal:
    score = _name_score(name, r.get('entity_name', ''))
    print(f"  [{score:.2f}] {r['entity_name']} | {r.get('uen_status_desc')}")

# Strategy B: full core name q= search
core = _normalize_name(name)
rows_core = _query({'q': core, 'limit': 20})
print(f'\nq="{core}" results:')
for r in rows_core:
    score = _name_score(name, r.get('entity_name', ''))
    print(f"  [{score:.2f}] {r['entity_name']} | postal: {r.get('reg_postal_code')}")


