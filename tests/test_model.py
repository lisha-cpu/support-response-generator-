import unittest
from src.ngram_model import NGramModel
from src.preprocess import tokenize, preprocess_corpus

class TestSupportResponseGenerator(unittest.TestCase):
    def setUp(self):
        sentences = preprocess_corpus([
            "hello how can i help",
            "hello thank you for contacting support",
            "your order has shipped"
        ])
        self.model = NGramModel(n=3, k=1.0).fit(sentences)

    def test_tokenize_lowercases(self):
        self.assertEqual(tokenize("Hello SUPPORT!"), ["hello", "support"])

    def test_suggestions_return_words_and_probabilities(self):
        results = self.model.suggestions(["hello"], limit=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(0 < probability <= 1 for _, probability in results))

    def test_generate_keeps_prompt(self):
        result = self.model.generate("hello", max_words=3)
        self.assertTrue(result.startswith("hello"))

if __name__ == "__main__":
    unittest.main()
