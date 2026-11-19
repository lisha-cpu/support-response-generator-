"""An n-gram language model with add-k (Laplace) smoothing."""
from collections import Counter, defaultdict
from typing import List, Tuple

class NGramModel:
    def __init__(self, n: int = 3, k: float = 1.0):
        if n < 2:
            raise ValueError("n must be at least 2")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.n = n
        self.k = k
        self.vocabulary = set()
        self.context_counts = Counter()
        self.ngram_counts = defaultdict(Counter)

    def fit(self, tokenized_sentences: List[List[str]]):
        """Learn counts of contexts and the words that follow them."""
        self.vocabulary = set()
        self.context_counts.clear()
        self.ngram_counts.clear()

        start = ["<s>"] * (self.n - 1)
        for sentence in tokenized_sentences:
            if not sentence:
                continue
            tokens = start + sentence + ["</s>"]
            self.vocabulary.update(tokens)
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                self.context_counts[context] += 1
                self.ngram_counts[context][word] += 1
        self.vocabulary.add("</s>")
        return self

    def suggestions(self, context: List[str], limit: int = 5) -> List[Tuple[str, float]]:
        """Return likely next words, using add-k smoothing."""
        if not self.vocabulary:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        context = [w.lower() for w in context]
        context = (["<s>"] * (self.n - 1) + context)[-(self.n - 1):]
        key = tuple(context)
        counts = self.ngram_counts.get(key, Counter())
        total = self.context_counts.get(key, 0)
        vocab_size = len(self.vocabulary)
        scored = [
            (word, (counts.get(word, 0) + self.k) /
             (total + self.k * vocab_size))
            for word in self.vocabulary
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def generate(self, prompt: str = "", max_words: int = 20) -> str:
        """Continue a prompt by repeatedly selecting the most likely next word."""
        from .preprocess import tokenize
        generated = tokenize(prompt)
        for _ in range(max_words):
            next_words = self.suggestions(generated, limit=1)
            if not next_words:
                break
            word = next_words[0][0]
            if word == "</s>":
                break
            generated.append(word)
        return " ".join(generated)
