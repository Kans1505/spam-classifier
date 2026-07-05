import streamlit as st
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

st.set_page_config(page_title="📧 Spam Classifier", layout="wide")
st.title("📧 Email Spam Classifier")
st.subheader("AI-Powered Spam Detection - 92% Accuracy")

with open('logistic_spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.write("---")
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Inbox", "📊 Stats", "ℹ️ About"])

with tab1:
    st.header("Analyze Your Gmail")
    if st.button("🔐 Connect Gmail"):
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        try:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        except:
            st.error("Gmail credentials not available on cloud. Use locally!")
            st.stop()
        
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=50).execute()
        messages = results.get('messages', [])
        
        st.success(f"Analyzing {len(messages)} emails...")
        
        spam_count = 0
        real_count = 0
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            features = [len(subject), subject.count('!'), subject.count('FREE')] + [0]*54
            pred = classifier.predict([features])[0]
            if pred == 1:
                spam_count += 1
                st.write(f"🚨 SPAM: {subject}")
            else:
                real_count += 1
                st.write(f"✅ REAL: {subject}")
        
        col1, col2 = st.columns(2)
        col1.metric("Real Emails", real_count)
        col2.metric("Spam Emails", spam_count)
with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "92.07%")
    col2.metric("Precision", "92.95%")
    col3.metric("Recall", "87.95%")

with tab3:
    st.write("GitHub: https://github.com/Kans1505/spam-classifier")

st.caption("🚀 Spam Classifier v1.0")