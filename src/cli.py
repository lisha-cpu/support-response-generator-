"""Interactive command-line interface for support response suggestions."""
from pathlib import Path
from .preprocess import preprocess_corpus, tokenize
from .ngram_model import NGramModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "support_corpus.txt"

def load_model():
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    sentences = preprocess_corpus(lines)
    return NGramModel(n=3, k=1.0).fit(sentences)

def main():
    model = load_model()
    print("\nSupport Response Generator")
    print("--------------------------")
    print("Type a customer message or response beginning.")
    print("Commands: 'quit' to exit, 'help' for instructions.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if prompt.lower() == "help":
            print("Enter a few words. The tool suggests likely next words and a continuation.")
            continue
        if not prompt:
            print("Please enter some text.\n")
            continue

        context = tokenize(prompt)
        suggestions = model.suggestions(context, limit=5)
        print("\nNext-word suggestions:")
        for word, probability in suggestions:
            print(f"  {word:<15} {probability:.3f}")

        continuation = model.generate(prompt, max_words=12)
        print(f"\nSuggested continuation: {continuation}\n")

if __name__ == "__main__":
    main()
