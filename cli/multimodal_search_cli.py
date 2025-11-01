import argparse
# Note: Assuming cli/lib/multimodal_search.py is available via standard Python import
# If you run the script from 'cli/', you might need to adjust the import path, 
# but for a standard module setup, the relative import below is often used.
# If the structure is strictly 'cli/lib/multimodal_search.py' and 'cli/multimodal_search_cli.py', 
# an absolute import from the project root might be necessary, but adhering to the common pattern:
from lib.multimodal_search import verify_image_embedding 


def main() -> None:
    """
    Parses command-line arguments and executes the corresponding multimodal search command.
    """
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Define the 'verify-image-embedding' command
    verify_parser = subparsers.add_parser(
        "verify_image_embedding", 
        help="Generates an embedding for an image and prints its dimensionality."
    )
    
    # Required positional argument for the image path
    verify_parser.add_argument(
        "image_path", 
        type=str, 
        help="The file path to the image to embed (e.g., path/to/image.jpg)"
    )

    args = parser.parse_args()

    # Dispatch based on the command
    match args.command:
        case "verify_image_embedding":
            # Call the top-level function from the library module
            verify_image_embedding(args.image_path)
            
        case _:
            # Prints help if no command or an invalid command is given
            if args.command is None:
                parser.print_help()
            else:
                print(f"Error: Unknown command '{args.command}'")
                parser.print_help()


if __name__ == "__main__":
    main()