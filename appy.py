import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")
st.title("📊 Social Media Sentiment Analyzer")
st.write("Sentiment analysis on social media text using NLP (VADER).")

# ---------------------------------
# DOWNLOAD VADER LEXICON (CLOUD SAFE)
# ---------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ---------------------------------
# LOAD DATA
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_tweets.csv")

df = load_data()

# ---------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------
def analyze_sentiment(data):
    analyzer = SentimentIntensityAnalyzer()

    data["Score"] = data["Tweet"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    def label(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    data["Sentiment"] = data["Score"].apply(label)
    return data

df = analyze_sentiment(df)

# ---------------------------------
# DISPLAY DATA
# ---------------------------------
st.subheader("📄 Tweet Data")
st.dataframe(df, use_container_width=True)

# ---------------------------------
# SENTIMENT BAR CHART
# ---------------------------------
st.subheader("📊 Sentiment Distribution")
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x="Sentiment",
        y="count()",
        color="Sentiment"
    )
)
st.altair_chart(chart, use_container_width=True)

# ---------------------------------
# WORD CLOUD
# ---------------------------------
st.subheader("☁ Word Cloud")
text = " ".join(df["Tweet"])
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white"
).generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---------------------------------
# FOOTER
# ---------------------------------
st.success("✅ App running successfully using NLTK VADER")
