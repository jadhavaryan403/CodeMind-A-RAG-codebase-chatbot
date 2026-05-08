from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from django.conf import settings
from utils.llm_callback import TokenTrackingCallback
import logging
import time

_explainer_chain = None
logger = logging.getLogger(__name__)


# 🔹 Structured Output Schema
class CodeExplanation(BaseModel):
    '''Pydantic model defining the expected structure of the code explanation output.'''
    one_line_summary: str = Field(description="Very short 5–10 word summary")
    detailed_explanation: str = Field(description="Full explanation of code")
    dependencies: list[str] = Field(
        description="List of function names used inside this code"
    )

def _get_explainer_chain():
    '''Initializes and returns a LangChain chain for generating code explanations.'''
    global _explainer_chain

    if _explainer_chain is None:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )

        parser = PydanticOutputParser(pydantic_object=CodeExplanation)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a senior software engineer.\n"
                "Analyze the given Python code and extract structured information.\n\n"
                "Rules:\n"
                "- one_line_summary: 5–10 words only\n"
                "- detailed_explanation: Explanation of the code of what it does in about 3-10 sentences based on it's size.\n"
                "- dependencies: ONLY non-built-in function/method names explicitly called\n"
                "- Do NOT hallucinate dependencies\n"
                "- Return valid JSON only\n\n"
                "{format_instructions}"
            ),
            (
                "human",
                "Symbol: {symbol_name} ({chunk_type})\n"
                "Lines {start_line}–{end_line}\n\n"
                "```python\n{code}\n```"
            ),
        ]).partial(format_instructions=parser.get_format_instructions())

        _explainer_chain = prompt | llm | parser

    return _explainer_chain


def generate_explanation(chunk) -> dict:
    '''Generates a structured explanation for a given code chunk using the LangChain chain.'''
    try:
        callback = TokenTrackingCallback()

        response = _get_explainer_chain().invoke(
            {
                "symbol_name": chunk.symbol_name,
                "chunk_type": chunk.chunk_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "code": chunk.code_text,
            },
            config={"callbacks": [callback]}
        )

        # ✅ Token usage
        print("Total tokens:", callback.total_tokens)
        print("Prompt tokens:", callback.prompt_tokens)
        print("Completion tokens:", callback.completion_tokens)

        return response.model_dump() ,callback

    except Exception as e:
        logger.error(f"Error generating explanation for {chunk.symbol_name}: {e}")
        return {
            "one_line_summary": f"{chunk.symbol_name} logic",
            "detailed_explanation": (
                f"{chunk.chunk_type} '{chunk.symbol_name}' "
                f"(lines {chunk.start_line}–{chunk.end_line})."
            ),
            "dependencies": [],
        }


def generate_explanations_batch(chunks: list) -> list[dict]:
    '''Generates explanations for a batch of code chunks, tracking token usage for each.'''
    results = []

    for chunk in chunks:
        try:
            res, callback = generate_explanation(chunk)

            results.append({
                "data": res,
                "usage": {
                    "prompt_tokens": callback.prompt_tokens,
                    "completion_tokens": callback.completion_tokens
                }
            })
            time.sleep(3)  # brief pause to avoid rate limits

        except Exception as e:
            logger.error(f"Error generating explanation for {chunk.symbol_name}: {e}")

            results.append({
                "data": {
                    "symbol_name": chunk.symbol_name,
                    "summary": f"{chunk.symbol_name} logic",
                    "dependencies": [],
                    "error": str(e)
                },
                "usage": None
            })

    return results