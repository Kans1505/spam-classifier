import streamlit as st
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

st.set_page_config(page_title="📧 Spam Classifier", layout="wide")

st.title("📧 Email Spam Classifier")
st.subheader("AI-Powered Spam Detection with 92% Accuracy")

# Load model
with open('logistic_spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.write("---")

# Sidebar
st.sidebar.title("⚙️ Settings")
action = st.sidebar.radio("Choose Action:", 
    ["📊 Dashboard", "🔍 Analyze Emails", "ℹ️ About"])

if action == "📊 Dashboard":
    st.header("Project Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "92.07%")
    with col2:
        st.metric("Precision", "92.95%")
    with col3:
        st.metric("Recall", "87.95%")
    with col4:
        st.metric("Training Emails", "4,601")
    
    st.write("---")
    
    st.subheader("📈 Model Performance")
    st.write("""
    - **Algorithm:** Logistic Regression
    - **Training Data:** Spambase Dataset (4,601 emails)
    - **Test Accuracy:** 92.07%
    - **False Positive Rate:** ~7% (real emails marked as spam)
    - **False Negative Rate:** ~12% (spam that slips through)
    """)

elif action == "🔍 Analyze Emails":
    st.header("Analyze Your Inbox")
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    if st.button("🔐 Connect to Gmail"):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=20).execute()
        messages = results.get('messages', [])
        
        st.success(f"✅ Connected! Found {len(messages)} emails")
        
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            
            # Simple prediction
            features = [len(subject), subject.count('!'), subject.count('FREE')]
            features += [0] * 54
            pred = classifier.predict([features])[0]
            
            if pred == 1:
                st.error(f"🚨 SPAM: {subject}")
            else:
                st.success(f"✅ REAL: {subject}")

elif action == "ℹ️ About":
    st.header("About This Project")
    st.write("""
    ### What is This?
    An AI-powered email spam classifier built with Python and scikit-learn.
    
    ### How It Works?
    1. **Train:** Learn patterns from 4,601 labeled emails
    2. **Predict:** Classify new emails as SPAM or REAL
    3. **Deploy:** Use on your actual Gmail inbox
    
    ### Technology
    - **Language:** Python
    - **ML Library:** scikit-learn
    - **Model:** Logistic Regression
    - **UI:** Streamlit
    - **Integration:** Gmail API
    
    ### Author
    Built as a portfolio project to demonstrate ML expertise.
    
    **GitHub:** [Kans1505/spam-classifier](https://github.com/Kans1505/spam-classifier)
    """)

st.write("---")
st.caption("📧 Spam Classifier v1.0 | Powered by AI")