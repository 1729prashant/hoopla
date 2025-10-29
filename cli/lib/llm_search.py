import os
from dotenv import load_dotenv
from google import genai

LLM_MODEL = "gemini-2.0-flash"

def llm_query_correction(query: str, enhance: str, usemodel: str = LLM_MODEL) -> str:
#===========================================================================================    
    spell_prompt_to_llm = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
#===========================================================================================
#===========================================================================================    
    rewrite_prompt_to_llm = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
#===========================================================================================
#===========================================================================================

    expand_prompt_to_llm = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
#===========================================================================================


    match enhance:
        case "spell":
            use_prompt = spell_prompt_to_llm
        case "rewrite": 
            use_prompt = rewrite_prompt_to_llm
        case "expand":
             use_prompt = expand_prompt_to_llm
        case _:
            use_prompt = rewrite_prompt_to_llm

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(model=usemodel, contents=use_prompt)

    return response.text



def llm_rerank(query: str, usemodel: str = LLM_MODEL) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(model=usemodel, contents=query)

    return response.text