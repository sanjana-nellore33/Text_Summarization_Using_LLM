
import os
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.5-flash-lite"


def get_gemini_client():
    """Create a Gemini client using the API key in .env."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please check your .env file."
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )


def generate_summary(client, prompt, max_output_tokens):
    """Send one summarization request to Gemini."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_output_tokens
        )
    )

    result = response.text

    if not result:
        raise ValueError(
            "Gemini returned an empty response. Please try again."
        )

    return result.strip()


def summarize_text(chunks, summary_length="Medium"):
    """Summarize short or long documents using Gemini."""

    if not chunks:
        raise ValueError("There is no document text to summarize.")

    client = get_gemini_client()

    length_instructions = {
        "Short": "Write approximately 100 words.",
        "Medium": "Write approximately 250 words.",
        "Detailed": "Write approximately 500 words."
    }

    instruction = length_instructions.get(
        summary_length,
        length_instructions["Medium"]
    )

    # Short documents need only one API request.
    if len(chunks) == 1:

        prompt = f"""
You are an expert document summarization assistant.

Summarize the following document.

Requirements:
- {instruction}
- Preserve important facts, names, dates and figures.
- Include the main ideas and conclusions.
- Do not invent information.
- Avoid unnecessary repetition.
- Use clear and understandable language.

DOCUMENT:
{chunks[0]}
"""

        return generate_summary(
            client,
            prompt,
            max_output_tokens=1000
        )

    # For long documents, summarize each chunk first.
    chunk_summaries = []

    for index, chunk in enumerate(chunks, start=1):

        prompt = f"""
Summarize document section {index}.

Requirements:
- Identify the important information.
- Preserve facts, names, dates and figures.
- Do not add unsupported information.
- Avoid unnecessary repetition.

DOCUMENT SECTION:
{chunk}
"""

        section_summary = generate_summary(
            client,
            prompt,
            max_output_tokens=700
        )

        chunk_summaries.append(section_summary)

    # Combine all section summaries into one final summary.
    combined_text = "\n\n".join(chunk_summaries)

    final_prompt = f"""
Create a complete document summary from the section summaries below.

Requirements:
- {instruction}
- Combine related ideas.
- Preserve important facts and the main message.
- Avoid repetition.
- Do not introduce unsupported information.
- Use clear and understandable language.

SECTION SUMMARIES:
{combined_text}
"""

    return generate_summary(
        client,
        final_prompt,
        max_output_tokens=1200
    )
