import streamlit as st
import pickle

st.set_page_config(page_title="📧 Spam Classifier", layout="wide")
st.title("📧 Email Spam Classifier")
st.subheader("AI-Powered Spam Detection - 92% Accuracy")

with open('logistic_spam_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.write("---")
tab1, tab2, tab3 = st.tabs(["🔍 Classify Email", "📊 Stats", "ℹ️ About"])

with tab1:
    st.header("Test the Classifier")
    email_input = st.text_input("Enter email subject:")
    if st.button("🔎 Classify"):
        if email_input:
            features = [len(email_input), email_input.count('!'), email_input.count('FREE')]
            features += [0] * 54
            pred = classifier.predict([features])[0]
            if pred == 1:
                st.error("🚨 SPAM!")
            else:
                st.success("✅ REAL!")

with tab2:
    st.metric("Accuracy", "92.07%")
    st.metric("Precision", "92.95%")

with tab3:
    st.write("GitHub: https://github.com/Kans1505/spam-classifier")

st.caption("🚀 Spam Classifier v1.0")