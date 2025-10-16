#!/usr/bin/env python3

import argparse
import json
import re
import nltk
import string
import pickle
import os
import pickle
import math
from collections import Counter
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
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc_id, text):
        """Tokenize text and add its tokens to the inverted index."""
        tokens = normalize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)

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
        with open("cache/term_frequencies.pkl", "wb") as f_term_frequencies:
            pickle.dump(self.term_frequencies, f_term_frequencies)

    def load(self):
        """load the index and docmap from disk using pickle."""
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl") or not os.path.exists("cache/term_frequencies.pkl"):
            raise FileNotFoundError("Cached index or docmap or term_frequencies not found. Please run `build` first.")

        with open("cache/index.pkl", "rb") as f_index:
            self.index = pickle.load(f_index)
        with open("cache/docmap.pkl", "rb") as f_docmap:
            self.docmap = pickle.load(f_docmap)
        with open("cache/term_frequencies.pkl", "rb") as f_term_frequencies:
            self.term_frequencies = pickle.load(f_term_frequencies)

    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = normalize_text(term)
        if len(tokens) != 1:
            raise ValueError("Expected exactly one token after normalization.")
        token = tokens[0]

        if doc_id not in self.term_frequencies:
            return 0

        return self.term_frequencies[doc_id].get(token, 0)
        # below is what is return statement is doing
#        if doc_id in self.term_frequencies:
#            if token in self.term_frequencies[doc_id]:
#                return self.term_frequencies[doc_id][token]
#            else:
#                return 0
#        else:
#            return 0
#        Look up the Counter for this document.
#        Then look up how many times this token appears in that Counter.
#        If the token isn’t there, return 0.


    def get_bm25_idf(self, term: str) -> float:
        """Compute BM25 IDF for a single normalized term."""
        N = len(self.docmap)
        
        normalized_term = normalize_text(term)
        if len(normalized_term) != 1:
            print("Error: BM25 IDF can only be computed for a single token.")
            return
        term = normalized_term[0]
        df = len(self.index.get(term, set()))

        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf




# ===============================
# BM25 FUNCTION
# ===============================

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        return
    
    bm25_idf_score = index.get_bm25_idf(term)
    
    return bm25_idf_score



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

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term in a document")
    idf_parser.add_argument("term", type=str, help="Term to look up")


    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

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
        
        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            tf_value = index.get_tf(args.doc_id, args.term)
            print(tf_value)

        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            doc_count = len(index.docmap)

            normalized_term = normalize_text(args.term)
            if len(normalized_term) != 1:
                print("Error: IDF can only be computed for a single token.")
                return
            term = normalized_term[0]
            term_doc_count = len(index.index.get(term, set()))  # count how many documents contain the term (the size of the set for that token)     
            
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            doc_count = len(index.docmap)

            normalized_term = normalize_text(args.term)
            if len(normalized_term) != 1:
                print("Error: IDF can only be computed for a single token.")
                return
            term = normalized_term[0]
            term_doc_count = len(index.index.get(term, set()))  # count how many documents contain the term (the size of the set for that token)     
            
            tf = index.get_tf(args.doc_id, args.term)
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            tfidf = tf*idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")


        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            if bm25idf is not None:
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
