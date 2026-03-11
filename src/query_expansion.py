from typing import List

from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY


def generate_query_variations(query: str, num_variations: int = 3) -> List[str]:
    """
    Generate alternative search query variations for a user query.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    prompt = f"""
You are helping improve document retrieval.

Generate {num_variations} alternative search queries for the question below.
Keep them concise and semantically related.
Return only the queries, one per line, without numbering.

Original question:
{query}
""".strip()
    
    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.split("\n") if line.strip()]

    variations = []
    for line in lines:
        cleaned = line.lstrip("- ").strip()
        if cleaned:
            variations.append(cleaned)

    return variations[:num_variations]