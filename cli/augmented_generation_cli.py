import argparse
import json
from dotenv import load_dotenv
from google import genai
import os
from lib.hybrid_search import HybridSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def load_movies():
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["movies"]


def gemini_prompt(query: str, docs) -> str:
    return f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""


def rag_command(query: str, limit: int, k = 60) -> None:

    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k, limit)

    print(f"Search Results:")
    for i, result in enumerate(results, start=1):
        print(f"  - {result['title']}")

    print(f"\nRAG Response:")
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=gemini_prompt(query, results)
    )
    print(response.text)



def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(args.query,5)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


