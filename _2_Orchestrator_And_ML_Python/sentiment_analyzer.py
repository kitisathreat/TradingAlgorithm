from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(self, text: str) -> float:
        if not text:
            return 0.0
        score = self.analyzer.polarity_scores(text)
        print(f"[Sentiment] Headline: '{text}' | Score: {score['compound']}")
        return score['compound']
