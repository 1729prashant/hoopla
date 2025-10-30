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
        relevant_retrieved = data['relevant_docs']
        relevant_retrieved_set = set(relevant_retrieved)

        results = searcher.rrf_search(query, k, limit)
        total_retrieved = [result["title"].strip() for result in results]
        total_retrieved_set = set(total_retrieved)

        # true_positives = len(total_retrieved.intersection(relevant_retrieved))

        precision = len(relevant_retrieved_set & total_retrieved_set) / len(total_retrieved)
        recall = len(relevant_retrieved_set & total_retrieved_set) / len(relevant_retrieved)
        
        if precision + recall == 0.0:
            f1_score = 0.0
        else:    
            f1_score = 2 * (precision * recall) / (precision + recall)

        precision_list.append({"query":     query, 
                               "precision": precision,
                               "recall": recall, 
                               "f1_score": f1_score,
                               "retrieved": ", ".join(total_retrieved), 
                               "relevant":  ", ".join(relevant_retrieved)})

    # precision_list = sorted(precision_list, key=lambda x: x["precision"], reverse=True)
    
    for q in precision_list:
        print(f"\n- Query: {q['query']}")
        print(f"   - Precision@{limit}: {q['precision']:.4f}")
        print(f"   - Recall@{limit}: {q['recall']:.4f}")
        print(f"   - F1 Score: {q['f1_score']:.4f}")
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
    evaluation_command(limit, k = 60 if limit < 5 else limit)



if __name__ == "__main__":
    main()
