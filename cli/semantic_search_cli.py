#!/usr/bin/env python3

import argparse
# from lib.semantic_search import verify_model
# from lib.semantic_search import embed
import lib.semantic_search as ll

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies that the model loads correctly")
    embed_parser = subparsers.add_parser("embed_text", help="use to pass text with embed_text <your text here>")
    embed_parser.add_argument("text", type=str, help="text")

    verify_embeddings = subparsers.add_parser("verify_embeddings", help="verifies embeddings")

    args = parser.parse_args()
    match args.command:
        
        case "verify":
            ll.verify_model()
        
        case "embed_text":
            ll.embed(args.text)

        case "verify_embeddings":
            ll.verify_embeddings()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


