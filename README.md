# Windows
cd "C:\Users\bruce.yee\...\capstone-project"
python setup.py
venv\Scripts\activate

# macOS
cd /path/to/capstone-project
python3 setup.py
source venv/bin/activate

After running setup.py
# Windows — note the quotes around the filename with spaces
python check.py --instructions "directives/Instructions (Directive).docx" --folder submissions/

# macOS
python check.py --instructions 'directives/Instructions (Directive).docx' --folder submissions/

new: 
docker compose build
docker compose run --rm checker
docker compose run --rm checker python check.py \
  --instructions "directives/Instructions (Directive).docx" \
  --document submissions/my_doc.pdf

  # Local Development 
  pip install pdfplumber python-docx pdf2image pytesseract Pillow
  cd "C:\Users\bruce.yee\OneDrive - Accenture\Documents\GitHub\capstone-project"
python check.py --instructions "directives/Main_MP-D 401-01-2005A Deferment Disruption and Exemption Policy for NSmen_050620.pdf" --folder submissions/ --output output/results.csv