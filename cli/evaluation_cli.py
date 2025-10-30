import argparse
import json 
from lib.hybrid_search import HybridSearch

def load_golden_dataset():
    with open('data/golden_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["test_cases"]

def load_movies():
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["movies"]


def evaluation_command(limit: int, k = 60) -> None:
    golden_dataset = load_golden_dataset()
    movies = load_movies()
    searcher = HybridSearch(movies)
    precision_list = []
    print(f"k={limit}")
    for i, data in enumerate(golden_dataset,start=0):
        query = data['query']
        relevant_retrieved = set(data['relevant_docs'])

        results = searcher.rrf_search(query, k, limit)
        total_retrieved = set([result["title"] for result in results])

        true_positives = len(total_retrieved.intersection(relevant_retrieved))

        precision = true_positives / len(total_retrieved)

        precision_list.append({"query":     query, 
                               "precision": precision, 
                               "retrieved": ", ".join(total_retrieved), 
                               "relevant":  ", ".join(relevant_retrieved)})

    # precision_list = sorted(precision_list, key=lambda x: x["precision"], reverse=True)

    for q in precision_list:
        print(f"\n- Query: {q['query']}")
        print(f"   - Precision@{limit}: {q['precision']:.4f}")
        print(f"   - Retrieved: {q['retrieved']}")
        print(f"   - Relevant: {q['relevant']}")

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    evaluation_command(limit)



if __name__ == "__main__":
    main()