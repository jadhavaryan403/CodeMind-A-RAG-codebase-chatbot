from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from django.conf import settings

_explainer_chain = None

def _get_explainer_chain():
    global _explainer_chain
    if _explainer_chain is None:
        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior software engineer. Given a Python code snippet, "
             "write a concise 2-5 sentence explanation of what it does, its purpose, "
             "key parameters, return values, and any important side effects. "
             "Be precise. Respond with plain prose only — no Markdown, no bullets."),
            ("human",
             "Symbol: {symbol_name} ({chunk_type})\n"
             "Lines {start_line}–{end_line}\n\n"
             "```python\n{code}\n```"),
        ])
        _explainer_chain = prompt | llm
    return _explainer_chain


def generate_explanation(chunk) -> str:
    response = _get_explainer_chain().invoke({
        "symbol_name": chunk.symbol_name,
        "chunk_type":  chunk.chunk_type,
        "start_line":  chunk.start_line,
        "end_line":    chunk.end_line,
        "code":        chunk.code_text,
    })
    return response.content.strip()


def generate_explanations_batch(chunks: list) -> list[str]:
    explanations = []
    for chunk in chunks:
        try:
            explanations.append(generate_explanation(chunk))
        except Exception:
            fallback = (
                f"{chunk.chunk_type} '{chunk.symbol_name}' "
                f"(lines {chunk.start_line}–{chunk.end_line})."
            )
            explanations.append(fallback)
    return explanations