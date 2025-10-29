import argparse
import json
import lib.hybrid_search as lh
import lib.llm_search as llm_lib

def load_movies():
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_scores_parser = subparsers.add_parser("normalize", help="normalizes list of scores")
    # Using '+' instead of '*' ensures args.scorelist is never None, only an empty list []
    normalize_scores_parser.add_argument("scorelist", nargs='+', type=str, help="List of scores (space-separated)")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="weighted search")
    weighted_search_parser.add_argument("query", type=str, help="term to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="weights between 0.0 (semantic) and 1.0 (keyword)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="reciprocal ranked search")
    rrf_search_parser.add_argument("query", type=str, help="term to search for")
    rrf_search_parser.add_argument("--k", type=float, default=60, help="default 60")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_search_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method",type=str,choices=["individual"],help="Query reranking method")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            if len(args.scorelist) == 0:
                print("Please provide scores to normalize.")
                return
            else:
                normalised_list = lh.normalize_scores(args.scorelist)
                for n_val in normalised_list:
                    print(f"* {n_val:.4f}")
        
        case "weighted-search":
            
            result = lh.weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            
            updated_query = args.query
            if args.enhance != None:
                updated_query = llm_lib.llm_query_correction(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{updated_query}'\n")

            rrf_search_results = lh.rrf_search_command(updated_query,args.k,args.limit, args.rerank_method)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

