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


def gemini_summarize_prompt(query: str, results) -> str:
    return f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

def rag_summarize_command(query: str, limit: int, k = 60) -> None:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k, limit)

    print(f"Search Results:")
    for i, result in enumerate(results, start=1):
        print(f"  - {result['title']}")

    print(f"\nRAG Response:")
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=gemini_summarize_prompt(query, results)
    )
    print(response.text)


def gemini_citations_prompt(query: str, results) -> str:
    return  f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""


def rag_citations_command(query: str, limit: int, k = 60) -> None:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k, limit)

    print(f"Search Results:")
    for i, result in enumerate(results, start=1):
        print(f"  - {result['title']}")

    print(f"\nLLM Answer:")
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=gemini_citations_prompt(query, results)
    )
    print(response.text)


def gemini_question_prompt(query: str, results) -> str:
    return f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{results}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

def rag_question_command(query: str, limit: int, k = 60) -> None:
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k, limit)

    print(f"Search Results:")
    for i, result in enumerate(results, start=1):
        print(f"  - {result['title']}")

    print(f"\nAnswer:")
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=gemini_question_prompt(query, results)
    )
    print(response.text)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_summarize_parser = subparsers.add_parser("summarize", help="Perform Summary RAG (search + generate answer)")
    rag_summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rag_citations_parser = subparsers.add_parser("citations", help="Perform Citations RAG (search + generate answer)")
    rag_citations_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_citations_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rag_question_parser = subparsers.add_parser("question", help="Perform Citations RAG (search + generate answer)")
    rag_question_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_question_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(args.query,5)
        
        case "summarize":
            rag_summarize_command(args.query, args.limit)

        case "citations":
            rag_citations_command(args.query, args.limit)

        case "question":
            rag_question_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


