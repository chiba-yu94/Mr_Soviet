import pandas as pd

pred = pd.read_csv("scripts/output.csv")
true = pd.read_csv("data/csv/group_labels.csv")
true.set_index("画像名", inplace=True)

correct = {"出題形式": 0, "解答形式": 0, "単元（新）": 0}

for _, row in pred.iterrows():
    if row["画像名"] not in true.index:
        continue
    true_row = true.loc[row["画像名"]]
    for col in correct:
        if row[col] == true_row[col]:
            correct[col] += 1

total = len(pred)
print("Accuracy:")
for k in correct:
    print(f"{k}: {correct[k]} / {total} ({correct[k] / total:.2%})")