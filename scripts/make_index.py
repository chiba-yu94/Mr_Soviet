import os
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import faiss
import pickle
from predict_utils import get_embedding

TEACHER_DIR = "data/teacher"
CSV_PATH = "data/csv/group_labels.csv"

labels_df = pd.read_csv(CSV_PATH)
labels_df.set_index("画像名", inplace=True)

embeddings = []
labels = []

for file in os.listdir(TEACHER_DIR):
    if file.endswith(".webp"):
        path = os.path.join(TEACHER_DIR, file)
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang="jpn")
        if file not in labels_df.index:
            continue
        row = labels_df.loc[file]
        emb = get_embedding(text)
        embeddings.append(emb)
        labels.append((file, row["出題形式"], row["解答形式"], row["単元（新）"]))

faiss_index = faiss.IndexFlatL2(EMBED_DIM)
faiss_index.add(np.array(embeddings))
faiss.write_index(faiss_index, "scripts/questions.index")
with open("scripts/labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("FAISS index built and saved.")