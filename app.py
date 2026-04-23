import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ------------------ Setup ------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ------------------ Load Model ------------------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ------------------ Text Processing ------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Spam Detector", page_icon="💎", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 15px;
    padding: 20px;
}

h1 {
    color: #38bdf8;
    text-align: center;
}

.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}

.spam {
    background-color: #7f1d1d;
    color: #fecaca;
}

.ham {
    background-color: #064e3b;
    color: #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<h1>💎 Spam Message Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>AI-powered spam detection system</p>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📌 About")
    st.write("""
    This app uses:
    - TF-IDF Vectorization  
    - Machine Learning Model  
    - NLP preprocessing  

    Built for spam detection in emails/SMS.
    """)

# ------------------ INPUT ------------------
input_sms = st.text_area("✉️ Enter your message", height=150)

# ------------------ BUTTON ------------------
if st.button("🚀 Analyze Message"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        with st.spinner("Analyzing... 🔍"):
            # Preprocess
            transformed_sms = transform_text(input_sms)

            # Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # Predict
            result = model.predict(vector_input)[0]

            # Confidence
            try:
                prob = model.predict_proba(vector_input)[0]
                confidence = max(prob)
            except:
                confidence = None

        # ------------------ RESULT ------------------
        if result == 1:
            st.markdown(
                '<div class="result-card spam">🚨 SPAM MESSAGE DETECTED</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card ham">✅ SAFE MESSAGE</div>',
                unsafe_allow_html=True
            )

        # ------------------ CONFIDENCE ------------------
        if confidence:
            st.progress(int(confidence * 100))
            st.write(f"Confidence: **{confidence:.2f}**")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
