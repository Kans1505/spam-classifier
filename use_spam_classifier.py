import pickle

print("Loading saved model...\n")

# Load the saved model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded!\n")

# Test on new emails
test_emails = [
    "Click here to WIN FREE MONEY NOW!!!",
    "Hi, how was your day?",
    "URGENT: Verify your account password immediately",
    "Coffee at 3pm tomorrow?",
    "LIMITED TIME OFFER - BUY NOW",
    "Thanks for attending the meeting"
]

print("Classifying emails:\n")
for email in test_emails:
    email_numbers = vectorizer.transform([email])
    prediction = classifier.predict(email_numbers)[0]
    result = "🚨 SPAM" if prediction == 1 else "✅ REAL"
    print(f"{result} | {email}\n")