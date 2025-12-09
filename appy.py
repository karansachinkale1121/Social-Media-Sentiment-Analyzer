import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")
st.title("📊 Social Media Sentiment Analyzer")
st.write("Rule-based sentiment analysis on social media text.")

# ---------------------------------
# LOAD DATA
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_tweets.csv")

df = load_data()

# ---------------------------------
# CUSTOM SENTIMENT ANALYZER
# ---------------------------------
positive_words = {
    "love", "good", "great", "excellent", "happy", "amazing", "awesome",
    "impressive", "positive", "success", "beneficial", "like", "best"
}

negative_words = {
    "bad", "terrible", "worst", "hate", "poor", "sad", "angry",
    "crash", "disappointing", "problem", "issues", "fail", "broken"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    return text.split()

def analyze_sentiment(data):
    scores = []
    labels = []

    for text in data["Tweet"]:
        words = clean_text(str(text))
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)

        score = pos - neg
        scores.append(score)

        if score > 0:
            labels.append("Positive")
        elif score < 0:
            labels.append("Negative")
        else:
            labels.append("Neutral")

    data["Score"] = scores
    data["Sentiment"] = labels
    return data

df = analyze_sentiment(df)

# ---------------------------------
# DISPLAY DATA
# ---------------------------------
st.subheader("📄 Tweet Data")
st.dataframe(df, use_container_width=True)

# ---------------------------------
# SENTIMENT DISTRIBUTION
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
st.success("✅ App running using custom rule-based sentiment analysis")

