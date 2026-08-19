# Adversarial training: fix the encoding weakness in the Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ---- 1. Load and split the data (same settings as before) ----
df = pd.read_csv("Modified_SQL_Dataset.csv").drop_duplicates()
X = df['Query'].astype(str)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---- 2. The encoding operator (the one that broke the detectors) ----


def op_encode(q):
    return q.replace(" ", "%20")


# ---- 3. Take malicious TRAINING queries and make encoded versions ----
mal_train = X_train[y_train == 1].tolist()
encoded_train = [op_encode(q) for q in mal_train]          # for TRAINING
# And a separate encoded set from TEST attacks, for honest evaluation
mal_test = X_test[y_test == 1].tolist()
encoded_test = [op_encode(q) for q in mal_test]            # for TESTING

# ---- 4. Build TWO training sets: normal, and normal + encoded ----
# Baseline training set (as before)
X_train_base = X_train.tolist()
y_train_base = y_train.tolist()

# Adversarially-augmented training set: add the encoded attacks (label 1)
X_train_adv = X_train.tolist() + encoded_train
y_train_adv = y_train.tolist() + [1] * len(encoded_train)

# ---- 5. Helper: train a RF and measure detection on encoded attacks + FP on benign ----


def train_and_evaluate(X_tr, y_tr, label):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
    Xtr = vec.fit_transform(X_tr)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xtr, y_tr)

    # detection rate on ENCODED test attacks (should be low before, high after)
    enc_v = vec.transform(encoded_test)
    enc_rate = np.mean(clf.predict(enc_v) == 1) * 100

    # false-positive rate on BENIGN test traffic (the cost we must watch)
    benign_test = X_test[y_test == 0].tolist()
    ben_v = vec.transform(benign_test)
    fp_rate = np.mean(clf.predict(ben_v) == 1) * 100

    print(f"{label}")
    print(f"   Detection on ENCODED attacks: {enc_rate:5.1f}%")
    print(f"   False positives on BENIGN:    {fp_rate:5.1f}%\n")


# ---- 6. Compare: before vs after adversarial training ----
print("=== BEFORE adversarial training (baseline RF) ===")
train_and_evaluate(X_train_base, y_train_base, "Baseline")

print("=== AFTER adversarial training (RF + encoded attacks) ===")
train_and_evaluate(X_train_adv, y_train_adv, "Adversarially trained")
