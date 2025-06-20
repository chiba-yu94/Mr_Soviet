import os
import pandas as pd
import pytesseract
from PIL import Image
from predict_utils import predict_labels

TEST_DIR = "data/test"
results = []

for file in os.listdir(TEST_DIR):
    if file.endswith(".webp"):
        path = os.path.join(TEST_DIR, file)
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang="jpn")
        出題形式, 解答形式, 単元 = predict_labels(text)
        results.append({
            "画像名": file,
            "出題形式": 出題形式,
            "解答形式": 解答形式,
            "単元（新）": 単元
        })

pd.DataFrame(results).to_csv("scripts/output.csv", index=False)
print("Prediction complete. Output saved to output.csv")