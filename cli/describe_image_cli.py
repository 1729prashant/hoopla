import argparse
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
import mimetypes


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

query_reqrite_prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

def main():
    parser = argparse.ArgumentParser(description="multimodal search")
    parser.add_argument("--image",type=str,help="the path to an image file")
    parser.add_argument("--query",type=str,help="a text query to rewrite based on the image")


    args = parser.parse_args()
    image_path = args.image
    query = args.query


    if not os.path.exists(image_path):
        print(f"Error: The path '{image_path}' does not exist.")
        sys.exit(1) 
    
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, 'rb') as f:
        image_content = f.read()

    parts = [
    query_reqrite_prompt,
    types.Part.from_bytes(data=image_content, mime_type=mime),
    query.strip(),
    ]
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash', contents=parts
        )
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens: {response.usage_metadata.total_token_count}")


    


if __name__ == "__main__":
    main()
