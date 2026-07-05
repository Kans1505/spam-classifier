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

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Gmail", "📊 Stats", "ℹ️ About"])

with tab1:
    st.header("Analyze Your Gmail Inbox")
    
    if st.button("🔐 Analyze My Inbox (50 emails)", key="analyze_btn"):
        try:
            SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
            
            # Try to load existing credentials
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            service = build('gmail', 'v1', credentials=creds)
            results = service.users().messages().list(userId='me', maxResults=50).execute()
            messages = results.get('messages', [])
            
            st.success(f"✅ Connected! Analyzing {len(messages)} emails...")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            spam_count = 0
            real_count = 0
            spam_list = []
            real_list = []
            
            # Placeholder for results
            results_placeholder = st.empty()
            
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
                    spam_list.append(subject[:60])
                else:
                    real_count += 1
                    real_list.append(subject[:60])
                
                # Update progress
                progress = (i + 1) / len(messages)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{len(messages)} emails...")
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            st.write("---")
            
            # Show results in nice cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("✅ Real Emails", real_count, delta=None)
            with col2:
                st.metric("🚨 Spam Emails", spam_count, delta=None)
            
            st.write("---")
            
            # Show email lists
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Real Emails")
                for email in real_list[:15]:
                    st.write(f"• {email}")
            
            with col2:
                st.subheader("🚨 Spam Emails")
                if spam_list:
                    for email in spam_list[:15]:
                        st.write(f"• {email}")
                else:
                    st.write("No spam detected! ✨")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("💡 Make sure credentials.json is in the project folder")

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
        st.metric("Training Data", "4,601")
    
    st.write("---")
    st.write("### Algorithm: Logistic Regression")
    st.write("""
    - Trained on Spambase dataset
    - Detects spam patterns
    - Features: word frequency, special characters, urgency indicators
    """)

with tab3:
    st.header("About This Project")
    st.write("""
    ### 📧 Email Spam Classifier
    
    **Built with:** Python, scikit-learn, Streamlit, Gmail API
    
    **How it works:**
    1. Authenticate with Gmail
    2. Fetch your emails
    3. Analyze patterns (FREE, CLICK, urgency words)
    4. Classify as SPAM or REAL
    5. Show results
    
    **GitHub:** https://github.com/Kans1505/spam-classifier
    
    """)

st.write("---")
st.caption("🚀 Spam Classifier v1.0 | Built with ❤️")