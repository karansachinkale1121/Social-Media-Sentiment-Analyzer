import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------
# APP TITLE
# ---------------------------
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")
st.title("📊 Real-Time Social Media Sentiment Analyzer")
st.write("Analyze sentiment from Twitter posts on any topic in real time.")

# ---------------------------
# USER INPUT
# ---------------------------
topic = st.text_input("Enter a topic or hashtag (e.g., 'AI', '#Budget2025')")

limit = st.slider("Number of tweets to fetch", 50, 500, 150)

# ---------------------------
# FUNCTION: FETCH TWEETS
# ---------------------------
def fetch_tweets(query, limit):
    tweets_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets_list.append([tweet.date, tweet.user.username, tweet.content])
    df = pd.DataFrame(tweets_list, columns=["Date", "User", "Tweet"])
    return df

# ---------------------------
# FUNCTION: SENTIMENT
# ---------------------------
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["Score"] = df["Tweet"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    def labeler(x):
        if x > 0.05:
            return "Positive"
        elif x < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Score"].apply(labeler)
    return df

# ---------------------------
# MAIN LOGIC
# ---------------------------
if st.button("Analyze"):
    if topic.strip() == "":
        st.error("Please enter a topic before analyzing.")
    else:
        st.info("Fetching tweets...")
        df = fetch_tweets(topic, limit)

        st.info("Analyzing sentiment...")
        df = analyze_sentiment(df)

        st.subheader("📄 Tweet Data")
        st.dataframe(df, use_container_width=True)

        # ---------------------------
        # SENTIMENT COUNT BAR CHART
        # ---------------------------
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

        # ---------------------------
        # WORD CLOUD
        # ---------------------------
        st.subheader("☁ Word Cloud")
        text = " ".join(df["Tweet"])
        wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""
---
Made with ❤️ using Streamlit, snscrape, and VADER Sentiment Analyzer.
""")
