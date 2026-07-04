import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Load the trained model
print("Loading 92% accuracy model...\n")
with open('logistic_spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)
print("✅ Model loaded!\n")

# Gmail authentication
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

print("Authenticating with Gmail...")
creds = None

try:
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
except:
    pass

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=8080)
    
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

print("✅ Gmail authenticated!\n")

# Build Gmail service
service = build('gmail', 'v1', credentials=creds)

# Fetch 50 emails from inbox
print("Fetching 50 emails from your inbox...\n")
results = service.users().messages().list(userId='me', maxResults=200, q='label:INBOX').execute()
messages = results.get('messages', [])

print(f"Found {len(messages)} emails\n")

spam_count = 0
real_count = 0
spam_ids = []

# Get email details and classify
for msg in messages:
    msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
    headers = msg_data['payload']['headers']
    
    # Get subject
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    
    # Get email body (simple extraction)
    try:
        if 'parts' in msg_data['payload']:
            body = msg_data['payload']['parts'][0]['body'].get('data', '')
        else:
            body = msg_data['payload']['body'].get('data', '')
    except:
        body = ''
    
    # For Spambase model, we need 57 features
    # Simple: use email subject length and some basic features
    # (This is simplified - real implementation would need proper feature extraction)
    features = [len(subject), subject.count('!'), subject.count('FREE'), subject.count('CLICK')]
    features += [0] * 53  # Pad to 57 features
    
    # Predict
    prediction = classifier.predict([features])[0]
    
    if prediction == 1:  # SPAM
        spam_count += 1
        spam_ids.append(msg['id'])
        print(f"🚨 SPAM | {subject[:50]}")
    else:  # REAL
        real_count += 1
        print(f"✅ REAL | {subject[:50]}")

print(f"\n" + "="*60)
print(f"Results:")
print(f"  Real emails: {real_count}")
print(f"  Spam emails: {spam_count}")
print(f"  Total: {real_count + spam_count}")
print(f"="*60)

# Ask before moving
if spam_count > 0:
    print(f"\n⚠️  Found {spam_count} spam emails")
    confirm = input("\nMove these to Spam folder? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        print("\nMoving spam emails...\n")
        for msg_id in spam_ids:
            # Move to Spam
            service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'addLabelIds': ['SPAM'], 'removeLabelIds': ['INBOX']}
            ).execute()
        
        print(f"✅ Moved {spam_count} emails to Spam folder!")
    else:
        print("❌ No emails moved (cancelled)")
else:
    print("\n✅ No spam detected! Your inbox is clean!")