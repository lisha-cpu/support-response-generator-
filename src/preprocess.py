"""Text cleaning and tokenization utilities."""
import re
from typing import List

def tokenize(text: str) -> List[str]:
    """Lowercase text and keep simple words and apostrophes as tokens."""
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text)

def preprocess_corpus(lines):
    """Convert non-empty corpus lines into token lists."""
    return [tokenize(line) for line in lines if line.strip()]
