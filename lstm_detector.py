import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and clean the data
df = pd.read_csv("Modified_SQL_Dataset.csv")
df = df.drop_duplicates()
X = df['Query'].astype(str)
y = df['Label']

# 2. Split into training and test sets (same settings as the Random Forest)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Turn queries into sequences of numbers (character-level)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 4. Pad every sequence to the same length
MAX_LEN = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

vocab_size = len(tokenizer.word_index) + 1

# 5. Build the LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LEN),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# 6. Train (with early stopping to avoid overfitting)
early_stop = EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train_pad, y_train,
          validation_split=0.1,
          epochs=10,
          batch_size=64,
          callbacks=[early_stop])

# 7. Evaluate
probs = model.predict(X_test_pad)
preds = (probs > 0.5).astype(int)
print(classification_report(y_test, preds))
