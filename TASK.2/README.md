# FAQ Chatbot (NLP + Cosine Similarity)

A simple chatbot that answers user questions by matching them against a set of
predefined FAQs, using NLP preprocessing and TF-IDF + cosine similarity.

## How this maps to the assignment

| Task requirement | Where it's done |
|---|---|
| Collect FAQs (Q&A pairs) | `faqs.json` — 15 sample FAQs for an e-commerce/online shopping support topic |
| Preprocess text with NLTK | `preprocess()` in `faq_chatbot.py` — tokenizes, lowercases, removes punctuation/stopwords, lemmatizes |
| Match user question to closest FAQ | `FAQChatbot.get_response()` — TF-IDF vectorization + cosine similarity |

## Setup

```bash
pip install -r requirements.txt
```

The first time you run it, NLTK will automatically download the small data
packages it needs (`punkt`, `stopwords`, `wordnet`) — this requires an
internet connection just for that first run.

## Run

```bash
python faq_chatbot.py
```

Then just type your question, e.g.:

```
You: how can i track my order
Bot: Go to 'My Orders' section in your account and click on the order you
     want to track. You'll see real-time tracking details including
     courier and expected delivery date.
```

Type `debug: <your question>` to see the top-3 closest FAQ matches and their
similarity scores — useful for tuning the `similarity_threshold` value in
`FAQChatbot.__init__`.

Type `quit`, `exit`, or `bye` to end the chat.

## Using your own FAQs

Replace the contents of `faqs.json` with your own topic's questions and
answers, in the same format:

```json
[
  { "question": "...", "answer": "..." }
]
```

You can also point the script at a different file:

```bash
python faq_chatbot.py my_faqs.json
```

## How the matching works

1. Every FAQ question is cleaned (lowercased, punctuation stripped),
   tokenized, stripped of stopwords, and lemmatized.
2. All cleaned FAQ questions are turned into TF-IDF vectors.
3. When the user asks something, it goes through the same cleaning +
   vectorizing process.
4. Cosine similarity is computed between the user's question vector and
   every FAQ question vector; the FAQ with the highest score is returned
   as the answer, as long as it clears a minimum similarity threshold
   (default `0.25`). Below that, the bot admits it doesn't have a match
   instead of guessing.

## Known limitation & possible extension

TF-IDF + cosine similarity is essentially smart keyword overlap — it
doesn't understand synonyms (e.g. "shipping" vs "delivery" are treated as
different words) or deeper intent. If your assignment wants an
**intent-based** approach instead (the other technique mentioned in the
task), a natural next step is to:

- Group FAQs into intent categories (e.g. `tracking`, `returns`, `payment`)
- Train a simple text classifier (e.g. Naive Bayes or Logistic Regression
  on TF-IDF features) to predict the intent of a user message
- Return a canned answer per predicted intent, or fall back to the
  cosine-similarity method within that intent's FAQ subset

That's a good "improvements" section to mention if your report/demo asks
for one.
