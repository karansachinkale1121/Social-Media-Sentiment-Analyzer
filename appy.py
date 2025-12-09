import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")
st.title("📊 Social Media Sentiment Analyzer")

st.info("⚠ Live scraping works locally. Streamlit Cloud uses sample data due to platform limits.")

# -----------------------
# SENTIMENT FUNCTION
# -----------------------
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["Score"] = df["Tweet"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    def label(x):
        if x > 0.05:
            return "Positive"
        elif x < -0.05:
            return "Negative"
        return "Neutral"

    df["Sentiment"] = df["Score"].apply(label)
    return df

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_tweets.csv")

df = load_data()
df = analyze_sentiment(df)

# -----------------------
# DISPLAY DATA
# -----------------------
st.subheader("📄 Tweet Data")
st.dataframe(df, use_container_width=True)

st.subheader("📊 Sentiment Distribution")
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(x="Sentiment", y="count()", color="Sentiment")
)
st.altair_chart(chart, use_container_width=True)

st.subheader("☁ Word Cloud")
text = " ".join(df["Tweet"])
wc = WordCloud(width=1600, height=800, background_color="white").generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig)

st.success("✅ App running successfully on Streamlit Cloud")

