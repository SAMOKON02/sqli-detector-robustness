# Evasion harness: disguise attacks and test both detectors' robustness
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1. Load the dataset and pull out the malicious queries
dataset = pd.read_csv("Modified_SQL_Dataset.csv").drop_duplicates()
malicious = dataset[dataset['Label'] == 1]['Query'].astype(str).tolist()
# Take a sample of 500 attacks to disguise (as your methodology specifies)
attacks = malicious[:500]
print(f"Loaded {len(attacks)} malicious payloads to disguise.")

# Step 2. Define the obfuscation operators
# Each function would disguis a payload while keeping it a valid attack.


def op_comment(q):
    return q.replace(" ", "/**/", 1) if " " in q else q


def op_case(q):           # randomise the capitalisation of keywords
    return "".join(c.upper() if i % 2 == 0 else c.lower()
                   for i, c in enumerate(q))


def op_whitespace(q):     # swapping normal spaces for extra whitespace
    return q.replace(" ", "  ")


def op_encode(q):         # url-style encode the spaces
    return q.replace(" ", "%20")


operators = {
    "comment": op_comment,
    "case": op_case,
    "whitespace": op_whitespace,
    "encoding": op_encode,
}

# Step 3. Load the two trained detectors
rf_model = joblib.load("rf_model.joblib")
rf_vec = joblib.load("rf_vectoriser.joblib")

lstm_model = load_model("lstm_model.keras")
with open("lstm_tokenizer.pkl", "rb") as f:
    lstm_tok = pickle.load(f)
MAX_LEN = 200

# Step 4. Helper functions: how many attacks does each detector catch?


def rf_detection_rate(payloads):
    v = rf_vec.transform(payloads)
    preds = rf_model.predict(v)
    return np.mean(preds == 1)      # fraction correctly flagged as attacks


def lstm_detection_rate(payloads):
    seq = lstm_tok.texts_to_sequences(payloads)
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    probs = lstm_model.predict(padded, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    return np.mean(preds == 1)


# Step 5. Baseline: detection on the ORIGINAL (undisguised) attacks
print("\n=== BASELINE (no disguise) ===")
print(f"Random Forest catches: {rf_detection_rate(attacks)*100:.1f}%")
print(f"LSTM catches:          {lstm_detection_rate(attacks)*100:.1f}%")

# Step 6. Test each operator and measure the drop ----
print("\n=== UNDER EVASION (per operator) ===")
for name, func in operators.items():
    disguised = [func(a) for a in attacks]
    rf_rate = rf_detection_rate(disguised) * 100
    lstm_rate = lstm_detection_rate(disguised) * 100
    print(f"{name:12s} | Random Forest: {rf_rate:5.1f}%  |  LSTM: {lstm_rate:5.1f}%")
