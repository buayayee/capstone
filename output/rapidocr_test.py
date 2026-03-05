import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
from PIL import Image
from rapidocr import RapidOCR

ocr = RapidOCR()
img = np.array(Image.open("submissions/case17/New Employment Local 1_Page_1.jpg"))
result = ocr(img)

print("type:", type(result))
print("attrs:", [a for a in dir(result) if not a.startswith("_")])

# try common attributes
for attr in ["txts", "text", "boxes", "scores", "elapse"]:
    val = getattr(result, attr, "N/A")
    if val != "N/A":
        print(f"result.{attr} (first 3):", val[:3] if hasattr(val, '__getitem__') else val)
