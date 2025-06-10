import pytest
import sys
# Add the project root to the Python path to allow imports
sys.path.insert(0, '..')
from sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def analyzer():
    """Provides a single instance of SentimentAnalyzer for all tests."""
    return SentimentAnalyzer()

def test_positive_sentiment(analyzer):
    """Tests that a clearly positive sentence gets a high positive score."""
    score = analyzer.get_sentiment_score("This is a wonderful and amazing success!")
    assert score > 0.5

def test_negative_sentiment(analyzer):
    """Tests that a clearly negative sentence gets a high negative score."""
    score = analyzer.get_sentiment_score("This is a terrible, horrible failure and a disaster.")
    assert score < -0.5

def test_neutral_sentiment(analyzer):
    """Tests that a neutral, factual sentence gets a score near zero."""
    score = analyzer.get_sentiment_score("The report was published today.")
    assert -0.1 < score < 0.1

def test_empty_or_none_input(analyzer):
    """Tests that empty or None input is handled gracefully and returns 0."""
    assert analyzer.get_sentiment_score("") == 0.0
    assert analyzer.get_sentiment_score(None) == 0.0
