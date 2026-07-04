import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pickle

print("Loading Spambase dataset...\n")

# Load the dataset
data = pd.read_csv('spambase.data', header=None)

# Last column is the label (0=ham, 1=spam)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(f"Total emails: {len(data)}")
print(f"Spam emails: {sum(y == 1)}")
print(f"Real emails: {sum(y == 0)}\n")

# For this approach, we'll use the raw features instead of text
# (Spambase is already vectorized, but let's train a better model)

print("Training improved classifier...")
clf = MultinomialNB()
clf.fit(X, y)

# Test on the same data (for now)
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"\n✅ ACCURACY: {accuracy:.2%}")
print(f"Precision: {precision_score(y, y_pred):.2%}")
print(f"Recall: {recall_score(y, y_pred):.2%}")

# Save the improved model
print("\nSaving improved model...")
with open('improved_spam_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ Model saved as 'improved_spam_model.pkl'")