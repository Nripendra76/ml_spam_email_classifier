import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ------------------ Download NLTK Resources ------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab")
    except:
        pass

# ------------------ Setup ------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ------------------ Load Model ------------------
@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model


tfidf, model = load_models()

# ------------------ Text Processing ------------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    words = []

    for token in tokens:
        if token.isalnum():
            words.append(token)

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(words)


# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Spam Detector",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>

body{
    background-color:#0f172a;
}

.main{
    background:linear-gradient(135deg,#1e293b,#0f172a);
    border-radius:15px;
    padding:20px;
}

h1{
    color:#38bdf8;
    text-align:center;
}

.stTextArea textarea{
    background-color:#1e293b !important;
    color:white !important;
    border-radius:10px !important;
}

.stButton>button{
    background:linear-gradient(90deg,#38bdf8,#6366f1);
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}

.result-card{
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:20px;
    margin-top:20px;
}

.spam{
    background:#7f1d1d;
    color:#fecaca;
}

.ham{
    background:#064e3b;
    color:#bbf7d0;
}

</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown(
    "<h1>Spam Message Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>AI-powered spam detection system</p>",
    unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("About")

    st.write("""
This application uses:

- TF-IDF Vectorization
- Machine Learning Classification
- Natural Language Processing (NLP)

It predicts whether an SMS or Email message is Spam or Safe.
""")

# ------------------ User Input ------------------
input_sms = st.text_area(
    "Enter your message",
    height=150
)

# ------------------ Prediction ------------------
if st.button("Analyze Message"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")

    else:

        with st.spinner("Analyzing..."):

            transformed_sms = transform_text(input_sms)

            vector_input = tfidf.transform([transformed_sms])

            prediction = model.predict(vector_input)[0]

            confidence = None

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(vector_input)[0].max()

        if prediction == 1:

            st.markdown(
                '<div class="result-card spam"><strong>SPAM MESSAGE DETECTED</strong></div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-card ham"><strong>SAFE MESSAGE</strong></div>',
                unsafe_allow_html=True
            )

        if confidence is not None:
            st.progress(int(confidence * 100))
            st.write(f"Confidence: **{confidence:.2%}**")

# ------------------ Footer ------------------
st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:gray;'>Built with Streamlit</p>",
    unsafe_allow_html=True
)
