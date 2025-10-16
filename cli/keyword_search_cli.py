#!/usr/bin/env python3

import argparse
import json
import re
import nltk
import string
import pickle
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ===============================
# TEXT NORMALIZATION PIPELINE
# ===============================
# Initialize NLP tools
stemmer = PorterStemmer()

try:
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        # Read words, strip whitespace, and create a set
        STOP_WORDS = set(word.strip() for word in f)
except FileNotFoundError:
    print("Error: data/stopwords.txt not found. Using an english set from nltk.corpus for stopwords.")
    STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def normalize_text(text: str) -> list[str]:
    """
    Normalize text by:
    1. Lowercasing
    2. Removing punctuation
    3. Tokenizing
    4. Removing stopwords
    5. Stemming or Lemming
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation (keep only alphanumeric and spaces)
    text = text.translate(str.maketrans('', '', string.punctuation)) # if following lesson.
    # text = re.sub(r'[^a-z0-9\s]', '', text)

    # 3. Tokenize
    tokens = text.split()

    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # PorterStemmer doesn’t collapse “faster” → “fast”, because “faster” is an adjective and the stemmer preserves “-er” in many cases.
    # To fix that, use a lemmatizer, which is aware of parts of speech.
    # 5. Stem
    stemming_tokens = [stemmer.stem(t) for t in tokens]
    # 5. Lemmas
    lemmas = []
    for t in tokens:        
        best_lemma = t
        for pos_tag in ['v', 'n', 'a', 'r', 's']:
            new_lemma = lemmatizer.lemmatize(t, pos=pos_tag)
            # We take the *shortest* lemma, which is usually the base form.
            #if len(new_lemma) < len(best_lemma):
            #    best_lemma = new_lemma
            # A simple take-the-last-one is also common since WordNet is consistent:
            best_lemma = new_lemma 
            
        lemmas.append(best_lemma)

    return stemming_tokens



# ===============================
# INVERTED INDEX CLASS
# ===============================
class InvertedIndex:
    
    def __init__(self):
        # token -> set(doc_ids)
        self.index: dict[str, set[int]] = {}
        # doc_id -> movie dict
        self.docmap: dict[int, dict] = {}


    def __add_document(self, doc_id, text):
        """Tokenize text and add its tokens to the inverted index."""
        # tokens = text.split()
        tokens = normalize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        """Return sorted list of doc_ids for the given token."""
        term = term.lower()
        return sorted(list(self.index.get(term, set())))
    
    def build(self, movies_data: dict):
        """Build the inverted index and document map."""
        for i, m in enumerate(movies_data["movies"], start=1):
            self.docmap[i] = m
            full_text = f"{m['title']} {m['description']}"
            self.__add_document(i, full_text)
    
    def save(self):
        """Save the index and docmap to disk using pickle."""
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f_index:
            pickle.dump(self.index, f_index)
        with open("cache/docmap.pkl", "wb") as f_docmap:
            pickle.dump(self.docmap, f_docmap)

    def load(self):
        """load the index and docmap from disk using pickle."""
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("Cached index or docmap not found. Please run `build` first.")

        with open("cache/index.pkl", "rb") as f_index:
            self.index = pickle.load(f_index)
        with open("cache/docmap.pkl", "rb") as f_docmap:
            self.docmap = pickle.load(f_docmap)



# ===============================
# SEARCH FUNCTION
# ===============================
def search_func(index: InvertedIndex, search_term: str = "") -> None:
    """Search using the inverted index instead of iterating over movies."""

    query_tokens = set(normalize_text(search_term))
    found_docs = set()

    for token in query_tokens:
        docs_for_token = index.get_documents(token)
        for doc_id in docs_for_token:
            found_docs.add(doc_id)
            if len(found_docs) >=5:
                break

    if not found_docs:
        print("--nothing found--")
        return
    
    for i, doc_id in enumerate(sorted(found_docs), start=1):
        movie = index.docmap[doc_id]
        print(f"{i}. [{doc_id}] {movie['title']}")






# ===============================
# BUILD FUNCTION
# ===============================
def build_index() -> None:
    """Build and save the inverted index to disk."""
    movies_json = 'data/movies.json'
    try:
        with open(movies_json, 'r', encoding='utf-8') as file:
            movies_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: {movies_json} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {movies_json}.")
        return

    index = InvertedIndex()
    index.build(movies_data)
    index.save()


# ===============================
# MAIN CLI ENTRYPOINT
# ===============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using token matching")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build and cache inverted index")

    args = parser.parse_args()

    match args.command:

        case "search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            print(f"Searching for: {args.query}")
            search_func(index, args.query)

        case "build":
            print("Building inverted index...")
            build_index()
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
