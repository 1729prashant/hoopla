#!/usr/bin/env python3

import argparse
import json
import re
import lib.semantic_search as ll

def load_movies():
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["movies"]

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


    chunking = subparsers.add_parser("chunk", help="chunk texts")
    chunking.add_argument("query", type=str, help="text to chunk")
    chunking.add_argument("--chunk-size", type=int, default=200, help="chunking size")
    chunking.add_argument("--overlap", type=int, default=0, help="overlap amount")


    semantic_chunk = subparsers.add_parser("semantic_chunk", help="chunk texts using semantic chunk")
    semantic_chunk.add_argument("query", type=str, help="text to chunk")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="chunking size")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="overlap amount") 


    embed_chunks = subparsers.add_parser("embed_chunks", help="generate embedded chunks")

    search_chunked = subparsers.add_parser("search_chunked", help="generate embedded chunks")
    search_chunked.add_argument("query", type=str, help="query text")
    search_chunked.add_argument("--limit", type=int, default=5, help="Number of results to return")




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
            documents = load_movies()
            llminilm_semantic_load.load_or_create_embeddings(documents)

            search_results = llminilm_semantic_load.search(args.query,args.limit)
            
            for i, result in enumerate(search_results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}\n")

        case "chunk":
            words = args.query.split()
            chunk_size = args.chunk_size
            overlap = args.overlap
            chunked_text = []

            step = chunk_size - overlap
            if step <= 0:
                print("Error: overlap must be smaller than chunk size.")
                return

            for i in range(0, len(words), step):
                group = words[i:i + chunk_size]
                chunk = " ".join(group)
                chunked_text.append(chunk)

            print(f"Chunking {len(args.query)} characters")
            for i, chunk_line in enumerate(chunked_text, start=1):
                print(f"{i}. {chunk_line}")


        case "semantic_chunk":
            chunked_sentences = ll.semantic_chunking(args.query, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.query)} characters")
            for i, chunk_line in enumerate(chunked_sentences, start=1):
                print(f"{i}. {chunk_line}")

        case "embed_chunks":
            llminilm_semantic_load = ll.ChunkedSemanticSearch()

            # Load movie data, and create embeddings
            documents = load_movies()
            chunk_embeddings = llminilm_semantic_load.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(chunk_embeddings)} chunked embeddings")


        case "search_chunked":
            documents = load_movies()
            llminilm_semantic_load = ll.ChunkedSemanticSearch()
            chunk_embeddings = llminilm_semantic_load.load_or_create_chunk_embeddings(documents)
            results = llminilm_semantic_load.search_chunks(args.query, args.limit)
            for i, chunk_line in enumerate(results, start=0):
                print(f"\n{i+1}. {results[i]['title']} (score: {results[i]['score']:.4f})")
                print(f"   {results[i]['description']}...")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


