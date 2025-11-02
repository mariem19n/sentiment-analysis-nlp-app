import os
import joblib
import numpy as np
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Sentiment Demo", page_icon="🎬", layout="centered")

st.title("🎬 Sentiment Analysis")
st.caption("Type a movie review and test your models ")


PIPELINE_PATH = "models/logreg_tfidf_lemma_pipeline_model8.joblib"
BEST_ESTIMATOR_PATH = "models/logreg_best_estimator_model7.joblib"
VECTORIZER_PATH = "models/logreg_tfidf_vectorizer_model7.joblib"


class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(w) for w in word_tokenize(reviews)]


# ----- Lazy load models -----
@st.cache_resource
def load_pipeline(path=PIPELINE_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_best_estimator(model_path=BEST_ESTIMATOR_PATH, vec_path=VECTORIZER_PATH):
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vect = joblib.load(vec_path)
        return model, vect
    return None, None

pipe_model = load_pipeline()
best_lr, tfidf_vect = load_best_estimator()

# ----- Model selector -----
options = []
if pipe_model is not None:
    options.append("Raw-Text Pipeline (model_8)")
if best_lr is not None and tfidf_vect is not None:
    options.append("Best Estimator + TF-IDF (model_7)")

if not options:
    st.error("No models found. Place your `.joblib` files next to this app and refresh.")
    st.stop()

model_choice = st.selectbox("Choose a model:", options, index=0)
st.write("")

# ----- Input area -----
default_text = (
    "I was impressed by the performances and the story. "
    "Some scenes were slow, but overall it was great."
)
text = st.text_area("Write a review:", value=default_text, height=160)

colA, colB = st.columns([1,1])
with colA:
    run_btn = st.button("Predict", type="primary")
with colB:
    explain = st.toggle("Show LIME explanation", value=False)

label_names = {0: "Negative", 1: "Positive"}

# ----- Utility: predict for each model type -----
def predict_with_pipeline(texts):
    proba = pipe_model.predict_proba(texts)
    pred = pipe_model.predict(texts)
    return pred, proba

def predict_with_best_estimator(texts):
    X = tfidf_vect.transform(texts)
    proba = best_lr.predict_proba(X)
    pred = best_lr.predict(X)
    return pred, proba

# ----- Run prediction -----
if run_btn:
    if not text.strip():
        st.warning("Please enter a review.")
        st.stop()

    if model_choice.startswith("Raw-Text Pipeline"):
        pred, proba = predict_with_pipeline([text])
    else:
        pred, proba = predict_with_best_estimator([text])

    p0, p1 = float(proba[0][0]), float(proba[0][1])
    st.subheader(f"Prediction: **{label_names[int(pred[0])]}**")
    st.write(f"**P(Negative)=** `{p0:.3f}`  |  **P(Positive)=** `{p1:.3f}`")

    st.progress(p1 if pred[0] == 1 else p0)

    # -----  LIME explanation for raw-text pipeline -----
    if explain:
        if model_choice.startswith("Raw-Text Pipeline"):
            try:
                from lime.lime_text import LimeTextExplainer
                explainer = LimeTextExplainer(class_names=[0, 1])
                exp = explainer.explain_instance(
                    text_instance=text,
                    classifier_fn=pipe_model.predict_proba,
                    num_features=15,
                    top_labels=2
                )
                # Render HTML explanation
                import streamlit.components.v1 as components
                components.html(exp.as_html(), height=500, scrolling=True)
            except ModuleNotFoundError:
                st.info("LIME is not installed. Add `lime` to requirements and reinstall.")
        else:
            st.info("LIME demo works with the **raw-text pipeline**. Select that model above.")

# ----- Sidebar: info -----
with st.sidebar:
    st.markdown("### Models loaded")
    st.write("✅ Pipeline" if pipe_model else "❌ Pipeline ")
    st.write("✅ Best estimator " if best_lr else "❌ Best estimator ")
    st.write("✅ TF-IDF vectorizer" if tfidf_vect else "❌ TF-IDF vectorizer")
    st.divider()
    st.markdown(
        """
        **Tips**
        - The Pipeline model works **directly on raw text**.
        - The Best Estimator expects **TF-IDF features**, so we load the exact saved vectorizer.
        - Turn on *LIME explanation* to highlight influential words.
        """
    )
