import argparse
from typing import Union, List, Dict, Any
from lib.multimodal_search import verify_image_embedding, image_search_command 


def print_search_results(results: List[Dict[str, Any]]) -> None:
    """
    Prints the search results in the requested formatted style.
    """
    if not results:
        print("No results found.")
        return

    for i, res in enumerate(results, 1):
        # Format the description to show the first part, avoiding printing the full text
        description_snippet = res['description'][:100].strip()
        if len(res['description']) > 100:
            description_snippet += "..."
            
        print(f"{i}. {res['title']} (similarity: {res['similarity_score']:.3f})")
        print(f"   {description_snippet}")
        print()


def main() -> None:
    """
    Parses command-line arguments and executes the corresponding multimodal search command.
    """
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Define the 'verify_image_embedding' command
    verify_parser = subparsers.add_parser("verify_image_embedding", help="Generates an embedding for an image and prints its dimensionality.")
    # Required positional argument for the image path
    verify_parser.add_argument("image_path", type=str, help="The file path to the image to embed (e.g., path/to/image.jpg)")

    # Define the 'image_search' command 
    image_search_parser = subparsers.add_parser("image_search", help="Finds movies similar to the uploaded image.")
    image_search_parser.add_argument("image_path", type=str, help="The file path to the query image.")
    image_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    args = parser.parse_args()

    # Dispatch based on the command
    match args.command:
        case "verify_image_embedding":
            # Call the top-level function from the library module
            verify_image_embedding(args.image_path)
        
        case "image_search":
                print(f"Searching for movies matching image at: {args.image_path}\n")
                results = image_search_command(args.image_path, args.limit)
                print_search_results(results)

        case _:
            # Prints help if no command or an invalid command is given
            if args.command is None:
                parser.print_help()
            else:
                print(f"Error: Unknown command '{args.command}'")
                parser.print_help()


if __name__ == "__main__":
    main()