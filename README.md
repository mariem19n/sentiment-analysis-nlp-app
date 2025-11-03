# 🧠 Sentiment Analysis NLP App

An end-to-end Natural Language Processing (NLP) project for **sentiment classification** on the IMDB dataset.  
It combines classic Machine Learning techniques with Explainable AI and an interactive **Streamlit** web app.

---

## 🚀 Overview

This project demonstrates the full NLP workflow:
- Text preprocessing (cleaning, tokenization, lemmatization)
- Feature extraction with **TF-IDF** and **CountVectorizer** (n-grams)
- Model training & evaluation using:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - AdaBoost
- Hyperparameter tuning with **GridSearchCV**
- Explainable AI (XAI) visualization of important features
- Streamlit app for real-time text sentiment prediction

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **NLP** | NLTK, Scikit-learn |
| **Model Persistence** | joblib |
| **Visualization** | Matplotlib |
| **Web App** | Streamlit |

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mariem19n/sentiment-analysis-nlp-app.git
   cd sentiment-analysis-nlp-app

2. **Create and activate a virtual environment**
   python -m venv .venv
  .venv\Scripts\activate       # on Windows
  source .venv/bin/activate    # on Mac/Linux

3. **Install dependencies**
   pip install -r requirements.txt

4. **Run the Streamlit app**
   streamlit run app.py

## 📊 Notebook Exploration

Open the notebook for step-by-step data analysis and model training:

  jupyter notebook notebooks/Sentiment_Analysis.ipynb

## 🧩 Explainable AI (XAI)

The app integrates interpretability techniques to:

-Highlight influential words contributing to predictions.

-Visualize how TF-IDF features affect sentiment classification.

-Provide model transparency for end users.



