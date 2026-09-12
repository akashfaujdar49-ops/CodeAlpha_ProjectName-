"""
FAQ Chatbot
-----------
A simple rule-based chatbot that answers user questions by matching them
against a set of predefined FAQs using NLP preprocessing + cosine similarity.

Pipeline:
  1. Load FAQ dataset (question-answer pairs).
  2. Preprocess text using NLTK (lowercase, tokenize, remove stopwords/punctuation, lemmatize).
  3. Vectorize all FAQ questions using TF-IDF.
  4. For a new user question, vectorize it the same way and compute cosine
     similarity against every FAQ question.
  5. Return the answer for the best match (if similarity is above a threshold).
"""

import json
import string
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. One-time NLTK setup (downloads required data if not already present)
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Lowercase, tokenize, remove punctuation/stopwords, and lemmatize."""
    text = text.lower().translate(PUNCT_TABLE)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(tok) for tok in tokens
        if tok.isalpha() and tok not in STOPWORDS
    ]
    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# 3. FAQ Chatbot class
# ---------------------------------------------------------------------------
class FAQChatbot:
    def __init__(self, faq_path: str, similarity_threshold: float = 0.25):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]

        # Preprocess every FAQ question once, up front
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF vectorizer on the FAQ questions
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

        self.similarity_threshold = similarity_threshold

    def get_response(self, user_query: str):
        """Return (answer, matched_question, similarity_score) for the best match."""
        processed_query = preprocess(user_query)

        if not processed_query.strip():
            return (
                "Could you rephrase your question with a few more words?",
                None,
                0.0,
            )

        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.similarity_threshold:
            return (
                "Sorry, I couldn't find a matching answer for that. "
                "Could you try rephrasing, or ask something else?",
                None,
                best_score,
            )

        return self.answers[best_idx], self.questions[best_idx], best_score

    def top_matches(self, user_query: str, k: int = 3):
        """Debug helper: show the top-k closest FAQ questions and their scores."""
        processed_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked = sorted(zip(self.questions, similarities), key=lambda x: -x[1])
        return ranked[:k]


# ---------------------------------------------------------------------------
# 4. Command-line chat loop
# ---------------------------------------------------------------------------
def main():
    faq_path = sys.argv[1] if len(sys.argv) > 1 else "faqs.json"
    bot = FAQChatbot(faq_path)

    print("=" * 60)
    print(" FAQ Chatbot (type 'quit' or 'exit' to stop)")
    print(" Tip: type 'debug: <question>' to see top matching scores")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! 👋")
            break

        if user_input.lower().startswith("debug:"):
            query = user_input.split(":", 1)[1].strip()
            for q, score in bot.top_matches(query):
                print(f"  [{score:.2f}] {q}")
            continue

        answer, matched_question, score = bot.get_response(user_input)
        print(f"Bot: {answer}")
        # Uncomment the next line if you want to see which FAQ it matched:
        # if matched_question:
        #     print(f"     (matched: \"{matched_question}\", score={score:.2f})")


if __name__ == "__main__":
    main()
