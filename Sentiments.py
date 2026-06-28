from datasets import load_dataset
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]
cleaned_texts = []
def clean_tweet(text):
    for text in texts:
        text = re.sub(r'http\S+', '', text) 
        text = re.sub(r'#', '', text)
        text = re.sub(r'@\w+', '', text) 
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.lower()
        text = text.strip()
        cleaned_texts.append(text)

# for i in range(5):
#     print(cleaned_texts[i])
#     print(labels[i])
#     print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

vectorizer=TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test) 

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf,y_train)

y_pred=model.predict(X_test_tfidf)
accuracy=accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred, target_names=["Negative","Neutral","Positive"]))

def predict_review(text):
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'#', '', text)
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.lower()
    text = text.strip()
    vector=vectorizer.transform([text])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    label=""
    if(prediction==0):
        label="Negative"
    elif(prediction==1):
        label="Neutral"
    else:
        label="Positive"
    confidence = probability[prediction]
    print(f"Sentiment  : {label}")
    print(f"Confidence : {confidence*100:.2f}%")

import joblib
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
