#!/usr/bin/env python3

import argparse
import json
import re
import nltk
import string 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

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


def search_func(search_term: str = "") -> None:
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

    query_tokens = set(normalize_text(search_term))
    movie_count = 0

    for movie in movies_data["movies"]:
        title_tokens = set(normalize_text(movie['title']))

        # If any normalized token from query appears in title
        if query_tokens & title_tokens:
            movie_count += 1
            print(f"{movie_count}. {movie['title']}")

    if movie_count == 0:
        print("--nothing found--")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using token matching")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_func(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
 #   # download NLTK stopwords once
 #   try:
 #       nltk.data.find('corpora/stopwords')
 #   except LookupError:
 #       nltk.download('stopwords')
    main()
