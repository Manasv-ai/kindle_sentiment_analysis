# **Sentiment Analysis Using Machine Learning**

This project implements an end-to-end **Sentiment Analysis** pipeline using classical NLP techniques and machine-learning algorithms. It processes raw text data, performs preprocessing (tokenization, stopword removal, lemmatization), converts text into numerical vectors (TF-IDF, Word2Vec), trains models, evaluates performance, and generates predictions.

---

## 🚀 **Project Overview**

The goal of this project is to classify text (reviews/messages) into positive or negative sentiment categories.
The notebook covers:

* Text cleaning & preprocessing
* Stopword removal, tokenization, lemmatization
* Feature extraction (TF-IDF & Average Word2Vec)
* Model building & evaluation
* Saving vectorizers & models
* Testing on new input

---

## 📂 **Project Structure**

```
sentiment_analysis/
│
├── sentiment_analysis.ipynb   # Main notebook
├── data/                       # Raw and cleaned datasets
├── models/                     # Saved model files (TF-IDF, Word2Vec, Classifier)
└── README.md                   # Documentation
```

---

## 🔧 **Technologies Used**

### **Python Libraries**

* **pandas**, **numpy** – Data handling
* **scikit-learn** – TF-IDF, ML models, evaluation
* **nltk** – Tokenization, stopwords, lemmatization
* **gensim** – Word2Vec embeddings
* **re** – Text cleaning

---

## 🧹 **Preprocessing Pipeline**

1. Remove punctuation and special characters
2. Convert to lowercase
3. Tokenize text
4. Remove stopwords
5. Lemmatize words
6. Join tokens back into cleaned sentences

---

## 🧠 **Feature Engineering**

The project uses two types of vector representations:

### **1️⃣ TF-IDF (Term Frequency – Inverse Document Frequency)**

* Converts text into numerical features based on importance of words
* Works well with classical ML models

### **2️⃣ Average Word2Vec**

* Creates embeddings for each word using trained Word2Vec model
* Averages all word vectors in a sentence to create a feature vector

---

## 🤖 **Models Used**

* Logistic Regression
* Random Forest Classifier
* Support Vector Machine
* Naive Bayes
  (Depending on your final selection inside the notebook)

Each model is trained, evaluated, and compared on accuracy, precision, recall, and F1-score.

---

## 📈 **Model Evaluation**

The notebook includes:

* Confusion matrix
* Classification report
* Accuracy scores
* Error analysis

These metrics help identify model performance and areas for improvement.

---

## 📝 **How to Run**

1. Clone or download the project
2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook sentiment_analysis.ipynb
```

4. Run each cell sequentially
5. Train the model and evaluate results

---

## 📦 **Future Improvements**

* Hyperparameter tuning
* Using transformer-based models (BERT, DistilBERT)
* Deployment using Flask/FastAPI
* Streamlit/Gradio user interface for live predictions

---

## 👤 **Author**

Manas Khatri
