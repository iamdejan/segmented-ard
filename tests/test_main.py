import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import hello


def test_hello_returns_correct_message() -> None:
    """Test that hello() returns the expected greeting message."""
    result = hello()
    assert result == "Hello world"


def test_hello_returns_string() -> None:
    """Test that hello() returns a string."""
    result = hello()
    assert isinstance(result, str)
