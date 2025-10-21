#!/usr/bin/env python3

import argparse
import json
import lib.semantic_search as ll

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies that the model loads correctly")
    
    embed_parser = subparsers.add_parser("embed_text", help="use to pass text with embed_text <your text here>")
    embed_parser.add_argument("text", type=str, help="text")

    verify_embeddings = subparsers.add_parser("verify_embeddings", help="verifies embeddings")

    query_embeddings = subparsers.add_parser("embedquery", help="query embedding, usage embedquery <your query here>")
    query_embeddings.add_argument("query", type=str, help="query text")

    search_embeddings = subparsers.add_parser("search", help="term you want to search for")
    search_embeddings.add_argument("query", type=str, help="query text")
    search_embeddings.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()
    match args.command:
        
        case "verify":
            ll.verify_model()
        
        case "embed_text":
            ll.embed(args.text)

        case "verify_embeddings":
            ll.verify_embeddings()

        case "embedquery":
            ll.embed_query_text(args.query)

        case "search":
            llminilm_semantic_load = ll.SemanticSearch()

            # Load movie data, and create embeddings
            with open('data/movies.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents = data["movies"]
            llminilm_semantic_load.load_or_create_embeddings(documents)

            search_results = llminilm_semantic_load.search(args.query,args.limit)
            i=1
            for i, result in enumerate(search_results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}\n")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


