from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

print("Creating training data...\n")

# Real emails
ham_emails = [
    "Hi! How are you doing?",
    "Your appointment is tomorrow at 3pm",
    "Thanks for your help on the project",
    "Meeting notes are attached",
    "Let's grab coffee this weekend",
    "Your flight is confirmed",
    "Invoice for services rendered"
]

# Spam emails
spam_emails = [
    "Click here to WIN FREE MONEY!!!",
    "URGENT: Verify your password NOW",
    "Congratulations! You won an iPhone!",
    "Get rich quick scheme",
    "Limited time offer - ACT NOW",
    "You have been selected!",
    "CLICK HERE for amazing deals"
]

all_emails = ham_emails + spam_emails
labels = [0]*len(ham_emails) + [1]*len(spam_emails)

print(f"Created {len(ham_emails)} real emails")
print(f"Created {len(spam_emails)} spam emails\n")

# Train the model
print("Training model...")
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(all_emails)

classifier = MultinomialNB()
classifier.fit(X, labels)

print("Model trained!\n")

# Save the model and vectorizer
print("Saving model...")
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved as 'spam_model.pkl'")
print("✅ Vectorizer saved as 'vectorizer.pkl'")