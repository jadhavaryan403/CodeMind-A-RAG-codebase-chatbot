from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from django.conf import settings
from concurrent.futures import ThreadPoolExecutor, as_completed

_explainer_chain = None


# 🔹 Structured Output Schema
class CodeExplanation(BaseModel):
    one_line_summary: str = Field(description="Very short 5–10 word summary")
    detailed_explanation: str = Field(description="Full explanation of code")
    dependencies: list[str] = Field(
        description="List of function names used inside this code"
    )

def _get_explainer_chain():
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
    try:
        response = _get_explainer_chain().invoke({
            "symbol_name": chunk.symbol_name,
            "chunk_type":  chunk.chunk_type,
            "start_line":  chunk.start_line,
            "end_line":    chunk.end_line,
            "code":        chunk.code_text,
        })

        return response.model_dump()

    except Exception as e:
        # 🔹 Safe fallback (VERY important)
        print(e)
        return {
            "one_line_summary": f"{chunk.symbol_name} logic",
            "detailed_explanation": (
                f"{chunk.chunk_type} '{chunk.symbol_name}' "
                f"(lines {chunk.start_line}–{chunk.end_line})."
            ),
            "dependencies": [],
        }


def generate_explanations_batch(chunks: list) -> list[dict]:
    print("Entered sequential explanation generation")
    print(f"Chunks received: {len(chunks)}")

    results = []

    for chunk in chunks:
        try:
            res = generate_explanation(chunk)
            results.append(res)

            print(f"Generated: {chunk.symbol_name}")
            print(f"Summary: {res['one_line_summary']}")
            print(f"Dependencies: {res['dependencies']}")

        except Exception as e:
            print(f"Error in {chunk.symbol_name}: {e}")

            # fallback to safe empty structure
            results.append({
                "symbol_name": chunk.symbol_name,
                "summary": f"{chunk.symbol_name} logic",
                "dependencies": [],
                "error": str(e)
            })

    return results