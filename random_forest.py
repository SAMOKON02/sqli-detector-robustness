# Classical SQL injection detector: TF-IDF + Random Forest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1. Loading the dataset from the csv
dataset = pd.read_csv("Modified_SQL_Dataset.csv")
print("Dataset shape Rows and columns:", dataset.shape)
print(dataset['Label'].value_counts())

# Step 2. Removing duplicates and separating inputs and labels
dataset = dataset.drop_duplicates()
queries = dataset['Query'].astype(str)
labels = dataset['Label']


# Step 3. Spliting into training sets and test sets, stratified keeps class balance
X_train, X_test, y_train, y_test = train_test_split(
    queries, labels, test_size=0.2, stratify=labels, random_state=42)

# Step 4. Convert each query into numbers using character-level TF-IDF
vectoriser = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
X_train_vec = vectoriser.fit_transform(X_train)
X_test_vec = vectoriser.transform(X_test)

# Step 5. Train the Random Forest detector with the training data
detector = RandomForestClassifier(random_state=42)
detector.fit(X_train_vec, y_train)

# Step 6. test on unseen data and print the scorecard
preds = detector.predict(X_test_vec)
print(classification_report(y_test, preds))

# Save the trained model and vectoriser so we can reuse them later
joblib.dump(detector, "rf_model.joblib")
joblib.dump(vectoriser, "rf_vectoriser.joblib")
print("Random Forest model and vectoriser saved.")
