
import pickle
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import webbrowser

# Load our trained model
print("Loading spam classifier model...\n")
with open('spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded!\n")

# Gmail authentication
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

print("Opening browser for Gmail login...")
print("If browser doesn't open, manually visit the URL shown\n")

flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=8080)

print("✅ Gmail authenticated!\n")

# Get emails
service = build('gmail', 'v1', credentials=creds)
results = service.users().messages().list(userId='me', maxResults=5).execute()
messages = results.get('messages', [])

emails = []
for msg in messages:
    txt = service.users().messages().get(userId='me', id=msg['id']).execute()
    headers = txt['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    emails.append(subject)

# Classify
print("Classifying your emails:\n")
for email in emails:
    email_numbers = vectorizer.transform([email])
    prediction = classifier.predict(email_numbers)[0]
    result = "🚨 SPAM" if prediction == 1 else "✅ REAL"
    print(f"{result} | {email}\n")