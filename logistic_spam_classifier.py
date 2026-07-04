import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

print("Loading Spambase dataset...\n")

data = pd.read_csv('spambase.data', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(f"Total emails: {len(data)}")
print(f"Spam emails: {sum(y == 1)}")
print(f"Real emails: {sum(y == 0)}\n")

# Split into train and test (better evaluation!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression...\n")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Test on TEST data (more honest!)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ ACCURACY: {accuracy:.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")

# Save model
with open('logistic_spam_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("\n✅ Model saved as 'logistic_spam_model.pkl'")