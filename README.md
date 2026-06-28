# Social Media Analysis using Machine Learning

## Overview

This project is a Machine Learning-based sentiment analysis application that predicts the sentiment of Twitter posts as **Negative**, **Neutral**, or **Positive**.

The application is trained using the **TweetEval Sentiment** dataset provided by Cardiff NLP through the Hugging Face Datasets library. It performs text preprocessing, converts tweets into TF-IDF feature vectors, and classifies them using a Logistic Regression model. A simple Streamlit interface allows users to enter a tweet and instantly view the predicted sentiment along with the model's confidence.

---

## Features

* Predicts tweet sentiment as:

  * Negative
  * Neutral
  * Positive
* Cleans Twitter-specific text:

  * Removes URLs
  * Removes user mentions
  * Removes hashtags symbols
  * Converts text to lowercase
* Uses TF-IDF Vectorization for feature extraction
* Logistic Regression classifier
* Interactive Streamlit web application
* Displays prediction confidence

---

## Model

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF Vectorizer
* Dataset: TweetEval Sentiment (Cardiff NLP)
* Train/Test Split: 80% / 20%

### Performance

* Accuracy: *Update with your final accuracy*
* F1 Score: *Update with your final F1 score (for example, 63% if that's your measured result)*

---

## Dataset

Dataset used:

**TweetEval – Sentiment Analysis**

Source: Cardiff NLP through the Hugging Face Datasets library.

The dataset contains labeled Twitter posts with three sentiment classes:

* Negative
* Neutral
* Positive

---

## Project Structure

```text
Twitter-Sentiment-Analysis-ML/
│── app.py
│── ml_sentiment.py
│── requirements.txt
│── README.md
└── .gitignore
```

Generated after training (not tracked by Git):

```text
sentiment_model.pkl
tfidf_vectorizer.pkl
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Live Demo

Streamlit App:

**Add your deployed Streamlit URL here**
```text
https://sentiment-analysis-using-machine-learning-7pj4ek2fjtoh27htjdgh.streamlit.app/
```

---

## Technologies Used

* Python
* Streamlit
* scikit-learn
* Hugging Face Datasets
* Joblib
* Regular Expressions (re)

---

## Future Improvements

* Support emoji-aware sentiment analysis
* Improve tweet preprocessing
* Experiment with SVM, Random Forest, and Naive Bayes
* Replace TF-IDF with transformer-based models such as BERT
* Deploy using Docker or cloud services

---

## Author

Arya Pradeep Khamayacha

Mini Project – Twitter Sentiment Analysis using Machine Learning
