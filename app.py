import joblib
import re
import streamlit as st

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_tweet(text):
        text = re.sub(r'http\S+', '', text) 
        text = re.sub(r'#', '', text)
        text = re.sub(r'@\w+', '', text) 
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.lower()
        text = text.strip()
        return text


st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter a Tweet")

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet first.")
    else:
        cleaned = clean_tweet(tweet)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        if prediction == 0:
            label = "Negative"
        elif prediction == 1:
            label = "Neutral"
        else:
            label = "Positive"

        confidence = probability[prediction] * 100

        st.subheader("Prediction")

        st.write(f"**Sentiment:** {label}")

        st.write(f"**Confidence:** {confidence:.2f}%")

        st.progress(int(confidence))
        if prediction == 0:
            label = "Negative"
            color = "red"
        elif prediction == 1:
            label = "Neutral"
            color = "orange"
        else:
            label = "Positive"
            color = "green"

        st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
        st.progress(int(confidence))
        probs = model.predict_proba(vector)[0]
        st.write(f"Negative: {probs[0]*100:.1f}%")
        st.write(f"Neutral: {probs[1]*100:.1f}%")
        st.write(f"Positive: {probs[2]*100:.1f}%")

    