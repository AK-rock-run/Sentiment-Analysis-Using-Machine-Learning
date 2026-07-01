import joblib
import pandas as pd
import re
import streamlit as st

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
TEXT_COLUMNS = ["tweet", "tweet_text", "text", "content", "full_text", "message", "body"]

def clean_tweet(text):
        text = re.sub(r'http\S+', '', text) 
        text = re.sub(r'#', '', text)
        text = re.sub(r'@\w+', '', text) 
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.lower()
        text = text.strip()
        return text

def find_text_column(columns):
    lower_columns = {str(column).lower(): column for column in columns}
    for column in TEXT_COLUMNS:
        if column in lower_columns:
            return lower_columns[column]
    return None

def predict_tweet_sentiment(text):
    cleaned = clean_tweet(str(text))
    vector = vectorizer.transform([cleaned])
    prediction = int(model.predict(vector)[0])
    probability = model.predict_proba(vector)[0]

    labels = {
        0: ("Negative", "red"),
        1: ("Neutral", "orange"),
        2: ("Positive", "green"),
    }
    label, color = labels[prediction]
    confidence = probability[prediction] * 100
    return label, color, confidence, probability


st.title("Twitter Sentiment Analysis")

tab_single, tab_batch = st.tabs(["Single Tweet", "Batch CSV"])

with tab_single:
    tweet = st.text_area("Enter a Tweet")

    if st.button("Analyze Sentiment"):
        if tweet.strip() == "":
            st.warning("Please enter a tweet first.")
        else:
            label, color, confidence, probs = predict_tweet_sentiment(tweet)

            st.subheader("Prediction")

            st.write(f"**Sentiment:** {label}")

            st.write(f"**Confidence:** {confidence:.2f}%")

            st.progress(int(confidence))
            st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
            st.write(f"Negative: {probs[0]*100:.1f}%")
            st.write(f"Neutral: {probs[1]*100:.1f}%")
            st.write(f"Positive: {probs[2]*100:.1f}%")

with tab_batch:
    uploaded_file = st.file_uploader("Upload a CSV with tweet text", type="csv")

    if uploaded_file is None:
        st.info("Use a text column such as tweet, tweet_text, text, content, full_text, message, or body.")
    else:
        batch_df = pd.read_csv(uploaded_file)
        text_column = find_text_column(batch_df.columns)

        if text_column is None:
            st.error("CSV must include a text column such as tweet, tweet_text, text, content, full_text, message, or body.")
        else:
            labels = []
            confidence_scores = []
            for value in batch_df[text_column].fillna("").astype(str):
                label, _color, confidence, _probs = predict_tweet_sentiment(value)
                labels.append(label)
                confidence_scores.append(round(confidence, 2))

            result_df = batch_df.copy()
            result_df["predicted_sentiment"] = labels
            result_df["prediction_confidence"] = confidence_scores

            st.success(f"Analyzed {len(result_df)} rows from `{text_column}`.")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Labeled CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "tweet_sentiment_predictions.csv",
                "text/csv",
            )
