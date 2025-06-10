import pytest
import sys
sys.path.insert(0, '..')
from data_fetcher import FMPFetcher

def test_get_analyst_ratings_processing(monkeypatch):
    """
    Tests the FMPFetcher's logic by mocking the requests.get call.
    This ensures our data processing works correctly without network dependency.
    """
    # 1. Define the fake data that our mock API will return
    mock_fmp_data = [{
        "symbol": "MSFT",
        "date": "2025-06-10",
        "ratingBuy": 20,
        "ratingOverweight": 10,
        "ratingHold": 5,
        "ratingUnderweight": 2,
        "ratingSell": 1,
        "ratingStrongSell": 0
    }]

    # 2. Create a mock response object that the 'requests.get' will return
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data
        
        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception("HTTP Error")

    # 3. Define the monkeypatch function to replace 'requests.get'
    def mock_get(*args, **kwargs):
        return MockResponse(mock_fmp_data, 200)

    # 4. Apply the patch: whenever 'requests.get' is called, call 'mock_get' instead
    monkeypatch.setattr("requests.get", mock_get)

    # 5. Run the test
    fetcher = FMPFetcher()
    result = fetcher.get_analyst_ratings("MSFT")

    # 6. Assert the results based on our processing logic
    # Expected buy ratings = 20 + 10 = 30
    # Expected total ratings = 30 + 5 + 2 + 1 = 38
    # Expected buy_ratio = 30 / 38
    assert result is not None
    assert "buy_ratio" in result
    assert result["buy_ratio"] == pytest.approx(30 / 38)
