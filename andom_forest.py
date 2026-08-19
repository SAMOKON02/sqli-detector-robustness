# Classical SQL injection detector: TF-IDF + Random Forest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load the dataset
df = pd.read_csv("Modified_SQL_Dataset.csv")
print("Rows and columns:", df.shape)
print(df['Label'].value_counts())

# 2. Clean the data
df = df.drop_duplicates()
X = df['Query'].astype(str)
y = df['Label']

# 3. Split into training and test sets (stratified keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Convert queries to numbers using character-level TF-IDF
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
X_train_v = vec.fit_transform(X_train)
X_test_v = vec.transform(X_test)

# 5. Train the Random Forest detector
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_v, y_train)

# 6. Evaluate
preds = clf.predict(X_test_v)
print(classification_report(y_test, preds))

# Save the trained model and vectoriser so we can reuse them later
joblib.dump(clf, "rf_model.joblib")
joblib.dump(vec, "rf_vectoriser.joblib")
print("Random Forest model and vectoriser saved.")
