# Adversarial training on the LSTM: fix the encoding weakness
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

MAX_LEN = 200

# ---- 1. Load and split (same settings as always) ----
df = pd.read_csv("Modified_SQL_Dataset.csv").drop_duplicates()
X = df['Query'].astype(str)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---- 2. The encoding operator (the disguise that broke the detectors) ----


def op_encode(q):
    return q.replace(" ", "%20")


# ---- 3. Build encoded attack sets: one for training, one for testing ----
mal_train = X_train[y_train == 1].tolist()
encoded_train = [op_encode(q) for q in mal_train]
mal_test = X_test[y_test == 1].tolist()
encoded_test = [op_encode(q) for q in mal_test]
benign_test = X_test[y_test == 0].tolist()

# ---- 4. A function that builds, trains and evaluates an LSTM ----


def build_train_evaluate(train_texts, train_labels, label):
    # tokenizer must be fit on THIS training set
    tok = Tokenizer(char_level=True)
    tok.fit_on_texts(train_texts)

    def encode_texts(texts):
        return pad_sequences(tok.texts_to_sequences(texts), maxlen=MAX_LEN)

    X_tr = encode_texts(train_texts)
    y_tr = np.array(train_labels)

    model = Sequential([
        Embedding(input_dim=len(tok.word_index) + 1, output_dim=32),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    early = EarlyStopping(monitor='val_loss', patience=2,
                          restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_split=0.1, epochs=10, batch_size=64,
              callbacks=[early], verbose=1)

    # detection on ENCODED test attacks
    enc_pred = (model.predict(encode_texts(
        encoded_test), verbose=0) > 0.5).astype(int)
    enc_rate = np.mean(enc_pred == 1) * 100
    # false positives on BENIGN test traffic
    ben_pred = (model.predict(encode_texts(
        benign_test), verbose=0) > 0.5).astype(int)
    fp_rate = np.mean(ben_pred == 1) * 100

    print(f"\n{label}")
    print(f"   Detection on ENCODED attacks: {enc_rate:5.1f}%")
    print(f"   False positives on BENIGN:    {fp_rate:5.1f}%\n")


# ---- 5. BEFORE: train on normal data only ----
print("=== BEFORE adversarial training (baseline LSTM) ===")
build_train_evaluate(X_train.tolist(), y_train.tolist(), "Baseline LSTM")

# ---- 6. AFTER: train on normal data + encoded attacks ----
print("=== AFTER adversarial training (LSTM + encoded attacks) ===")
build_train_evaluate(
    X_train.tolist() + encoded_train,
    y_train.tolist() + [1] * len(encoded_train),
    "Adversarially trained LSTM"
)
