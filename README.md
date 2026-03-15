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

# Backend-only supporting document extraction (no frontend)
# Reads supporting docs from a folder and writes text + markdown + layout JSON
python preprocessing/extract_supporting_docs.py \
  --input-folder submissions \
  --output-folder output/supporting_docs \
  --recursive

# ── DistilBERT Training (no API key needed) ──────────────────────────────────
# Step 1+2+3: Seed genuine docs from submissions/, extract text, generate synthetic fraudulent samples
python prepare_training_data.py
# Optional flags:
#   --multiplier 5        how many synthetic fraudulent docs per genuine doc (default: 5)
#   --skip-seed           skip copying from submissions/ if data/genuine/ is already populated

# Step 4: Fine-tune DistilBERT (reads data/labeled/dataset_augmented.csv)
python training/train.py
# Model saved to: models/distilbert-fraud-classifier/

# Step 5: Start the inference API
uvicorn api.main:app --reload --port 8000

  # Local Development 
  pip install pdfplumber python-docx pdf2image pytesseract Pillow
  cd "C:\Users\bruce.yee\OneDrive - Accenture\Documents\GitHub\capstone-project"
python check.py --instructions "directives/Main_MP-D 401-01-2005A Deferment Disruption and Exemption Policy for NSmen_050620.pdf" --folder submissions/ --output output/results.csv


C:\cap_venv\Scripts\activate

# 1. Install deps (if not already)
pip install -r requirements.txt

# 2. Prepare training data (seeds genuine docs, extracts text, generates synthetic fraud samples)
python prepare_training_data.py

prepare_training_data.py --skip-seed

# 3. Fine-tune DistilBERT (CPU takes ~10-20 min for 14 docs × 5 multiplier = ~84 samples)
python training/train.py

# 4. Start the API (model will now load from models/distilbert-fraud-classifier/)
uvicorn api.main:app --reload --port 8000

API KEY 
$env:OPENAI_API_KEY = "sk-..."


$env:AWS_ACCESS_KEY_ID="AKIA..."
$env:AWS_SECRET_ACCESS_KEY="your-secret-key-here"
$env:AWS_DEFAULT_REGION="ap-southeast-1"