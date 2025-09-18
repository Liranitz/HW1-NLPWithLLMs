# --- Imports ---
import os
from google import genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum

# 
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"      # adjective
    ADP = "ADP"      # adposition
    ADV = "ADV"      # adverb
    AUX = "AUX"      # auxiliary verb
    CCONJ = "CCONJ"  # coordinating conjunction
    DET = "DET"      # determiner
    INTJ = "INTJ"    # interjection
    NOUN = "NOUN"    # noun
    NUM = "NUM"      # numeral
    PART = "PART"    # particle
    PRON = "PRON"    # pronoun
    PROPN = "PROPN"  # proper noun
    PUNCT = "PUNCT"  # punctuation
    SCONJ = "SCONJ"  # subordinating conjunction
    SYM = "SYM"      # symbol
    VERB = "VERB"    # verb
    X = "X"          # other

# TODO Define more Pydantic models for structured output
class TokenPOS(BaseModel):
    text: str = Field(description="The token text.")
    pos_tag: UDPosTag = Field(description="The Universal Dependencies POS tag for the token.")

class SentencePOS(BaseModel):
    tokens: List[TokenPOS] = Field(description="A list of tokens with their POS tags for this sentence.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    # genai.configure(api_key=api_key)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Concise prompt
    prompt = f"""
Tag each token in the following sentence(s) with its Universal Dependencies POS tag (one of: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X). Return a JSON object with this structure: {{'sentences': [{{'tokens': [{{'text': <token>, 'pos_tag': <UD POS tag>}}, ...]}}]}}. Sentence(s): {text_to_tag}
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )
    print(response.text)
    res: TaggedSentences = response.parsed
    return res


# --- Example Usage ---
if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    # example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"
    example_text = "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    print(f"\nTagging text: \"{example_text}\"")

    tagged_result = tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            for token_obj in s.tokens:
                token = token_obj.text
                tag = token_obj.pos_tag
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")