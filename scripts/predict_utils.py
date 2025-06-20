import openai
import numpy as np
import faiss
import pickle
import os

EMBED_DIM = 1536

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index + label dictionaries
index = faiss.read_index("scripts/questions.index")
with open("scripts/labels.pkl", "rb") as f:
    labels_list = pickle.load(f)  # list of tuples: (image_name, 出題形式, 解答形式, 単元)


def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response['data'][0]['embedding'], dtype="float32")


def predict_labels(text):
    vec = np.array([get_embedding(text)])
    _, I = index.search(vec, k=5)
    top = [labels_list[i] for i in I[0]]
    # majority voting for each label field
    def majority(n): return max(set(x[n] for x in top), key=lambda k: [x[n] for x in top].count(k))
    return majority(1), majority(2), majority(3)  # 出題形式, 解答形式, 単元