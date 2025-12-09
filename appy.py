import streamlit as st
import pandas as pd
from textblob import TextBlob
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")
st.title("📊 Social Media Sentiment Analyzer")
st.write("Sentiment analysis on social media text using TextBlob NLP.")

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
    def get_polarity(text):
        return TextBlob(str(text)).sentiment.polarity

    data["Score"] = data["Tweet"].apply(get_polarity)

    def label(score):
        if score > 0:
            return "Positive"
        elif score < 0:
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
wc = WordCloud(
    width=1600,
    height=800,
    background_color="white"
).generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---------------------------------
# FOOTER
# ---------------------------------
st.success("✅ App running successfully using TextBlob (cloud-safe)")
