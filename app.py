import streamlit as st
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

st.set_page_config(page_title="📧 Spam Classifier", layout="wide")

st.title("📧 Email Spam Classifier")
st.subheader("AI-Powered Spam Detection - 92% Accuracy")

# Load model
with open('logistic_spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.write("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Your Inbox", "📊 Stats", "ℹ️ About"])

with tab1:
    st.header("Analyze Your Gmail Inbox")
    st.write("Connect with your Gmail account to analyze your emails!")
    
    if st.button("🔐 Connect Gmail & Analyze", key="gmail_btn"):
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=50).execute()
        messages = results.get('messages', [])
        
        st.success(f"✅ Connected! Analyzing {len(messages)} emails...")
        
        spam_count = 0
        real_count = 0
        spam_list = []
        real_list = []
        
        progress_bar = st.progress(0)
        
        for i, msg in enumerate(messages):
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            
            # Simple features
            features = [len(subject), subject.count('!'), subject.count('FREE')]
            features += [0] * 54
            pred = classifier.predict([features])[0]
            
            if pred == 1:
                spam_count += 1
                spam_list.append(subject)
            else:
                real_count += 1
                real_list.append(subject)
            
            progress_bar.progress((i + 1) / len(messages))
        
        st.write("---")
        st.write("## Results:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real Emails", real_count, delta=None)
        with col2:
            st.metric("Spam Emails", spam_count, delta=None)
        
        st.write("---")
        st.write("### ✅ Real Emails:")
        for email in real_list[:10]:
            st.write(f"• {email}")
        
        if spam_count > 0:
            st.write("### 🚨 Spam Emails:")
            for email in spam_list[:10]:
                st.write(f"• {email}")

with tab2:
    st.header("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "92.07%")
    with col2:
        st.metric("Precision", "92.95%")
    with col3:
        st.metric("Recall", "87.95%")
    with col4:
        st.metric("Trained On", "4,601 emails")

with tab3:
    st.header("About")
    st.write("""
    Built with Python & scikit-learn
    GitHub: https://github.com/Kans1505/spam-classifier
    """)

st.write("---")
st.caption("🚀 Spam Classifier v1.0")
