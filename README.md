# Support Response Generator

A beginner-friendly command-line tool that suggests likely next words and drafts short continuations for common customer-support messages. It uses a trigram (3-gram) language model with add-k/Laplace smoothing.

## Features

- Fictional, anonymized-style support corpus (no real customer data)
- Lowercase text preprocessing and tokenization
- Trigram language model
- Add-k smoothing to give unseen next words a non-zero probability
- Next-word suggestions with probabilities
- Simple interactive CLI
- Unit tests

## Requirements

- Python 3.9 or newer
- No third-party packages required

## Run on Windows

Open Command Prompt in the project folder and run:

```bash
python -m src.cli
```

If `python` is not recognized, try:

```bash
py -m src.cli
```

## Example

```text
Support Response Generator
--------------------------
Type a customer message or response beginning.
Commands: 'quit' to exit, 'help' for instructions.

You: your order
Next-word suggestions:
  has             0.XXX
  ...

Suggested continuation: your order ...
```

The exact suggestions depend on the corpus counts and vocabulary.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
support-response-generator/
├── data/
│   └── support_corpus.txt
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── ngram_model.py
│   └── preprocess.py
├── tests/
│   └── test_model.py
└── README.md
```

## How it works

1. **Corpus:** `data/support_corpus.txt` contains fictional support-response examples. Do not add real customer conversations unless they have been properly anonymized and you have permission to use them.
2. **Preprocessing:** `tokenize()` lowercases text and extracts words and numbers.
3. **N-gram training:** The model counts each two-word context and the word that follows it (trigrams).
4. **Smoothing:** Add-k smoothing calculates:

   `P(word | context) = (count(context, word) + k) / (count(context) + k * vocabulary_size)`

   The default `k` is 1.0. This prevents unseen words from receiving zero probability.
5. **Generation:** The generator repeatedly selects the highest-probability next word until it reaches the word limit or an end-of-sentence token.
6. **CLI:** Agents enter a prompt and see five next-word suggestions plus a short generated continuation.

## Important limitation

This is a small educational n-gram demo, not a production customer-support system. It selects the most likely next word greedily, so it can repeat or produce awkward text. A larger, approved corpus and human review would be needed for real customer use.

## Milestones

- [x] Prepare a fictional anonymized-style support corpus.
- [x] Preprocess and tokenize the corpus.
- [x] Implement an n-gram model with add-k smoothing.
- [x] Create a text generation function.
- [x] Build an interactive command-line interface.
