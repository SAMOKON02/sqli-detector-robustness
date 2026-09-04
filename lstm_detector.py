# Deep-learning SQL injection detector: character-level LSTM
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Step 1. read the csv file and remove duplicates
dataset = pd.read_csv("Modified_SQL_Dataset.csv")
dataset = dataset.drop_duplicates()
queries = dataset['Query'].astype(str)
labels = dataset['Label']

# Step 2. Spliting into training and testing sets with a 80/20 split(same settings as the Random Forest)
X_train, X_test, y_train, y_test = train_test_split(
    queries, labels, test_size=0.2, stratify=labels, random_state=42)

# Step 3. Turning each query into sequences of numbers (character level)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

# Step 4. Paddiin every sequence to the same length
MAX_LENGTH = 200
X_train_pad = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
X_test_pad = pad_sequences(test_sequences, maxlen=MAX_LENGTH)

vocab_size = len(tokenizer.word_index) + 1

# Step 5. Build the LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# Step 6. Train with early stopping to avoid overfitting
early_stop = EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train_pad, y_train,
          validation_split=0.1,
          epochs=10,
          batch_size=64,
          callbacks=[early_stop])

# Step 7. Evaluate
probs = model.predict(X_test_pad)
preds = (probs > 0.5).astype(int)
print(classification_report(y_test, preds))

# Save the trained LSTM model and its tokenizer so we can reuse them later
model.save("lstm_model.keras")
with open("lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("LSTM model and tokenizer saved.")
