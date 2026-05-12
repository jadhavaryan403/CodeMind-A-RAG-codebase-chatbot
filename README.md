# CodeMind - Codebase RAG Assistant

A **code-aware Retrieval-Augmented Generation (RAG)** system for querying and understanding software repositories using natural language. The system indexes source code into semantic chunks, generates explanations and dependency metadata, stores embeddings in FAISS, and answers developer questions with adaptive retrieval and multi-hop dependency fetching.

---

## Overview

Codebase RAG Assistant helps developers ask questions like:

* **Is the `register` function implemented?**
* **How is resume data parsed?**
* **What does this function return?**
* **Which functions does this module depend on?**

Instead of keyword search, the system builds a semantic understanding of the codebase and retrieves relevant implementation details.

---

# Features

## 1. AST-Based Code Chunking

Source files are parsed using Python Abstract Syntax Tree (AST) and split into logical chunks such as:

* Functions
* Classes
* Methods
* Important module-level code

Each chunk stores:

* Symbol name (Function/Class name = Symbol)
* File path
* Chunk type
* Start/end line numbers
* Raw code

**Benefit:** Retrieval happens at function/class level instead of entire files.

---

## 2. LLM-Based Code Explanation Generation

Each chunk is sent to an LLM to generate:

* **Detailed explanation** (Explanation of about 5-8 sentences used for embedding)
* **One-line summary** (short semantic label for multi dependency awareness)
* **Dependency list** (functions/classes used by the chunk)

Example:

```text
Function: handle_resume_upload
Summary: Processes uploaded PDF resume and recommends internships ...
Dependencies: [extract_text_from_pdf, clean_extracted_text, find_matching_internships]
```

**Benefit:** Embeddings are created from explanations rather than raw code, drastically improving semantic retrieval quality.

---

## 3. Dependency Graph Construction

Dependency metadata is extracted during explanation generation.

Example:

```text
upload_resume → handle_resume_upload
handle_resume_upload → extract_text_from_pdf
handle_resume_upload → clean_extracted_text
```

**Benefit:** Enables graph-aware retrieval and multi-hop reasoning.

---

## 4. FAISS Vector Store Indexing

Chunk explanations are embedded and stored in a **FAISS** vector database.

Stored metadata includes:

* Code snippet
* Symbol name
* Summary
* Dependencies
* Dependency summaries
* File path
* Line numbers

**Benefit:** Fast semantic similarity search over large codebases.

---

## 5. Incremental Re-indexing

The system stores SHA256 hashes of indexed chunks in Relational Database.

When files change:

* Unchanged chunks are skipped
* Only modified chunks are reprocessed
* FAISS index is updated efficiently

**Benefit:** Massive reduction in re-indexing time.

<img width="1536" height="1024" alt="ChatGPT Image May 12, 2026, 01_58_04 PM" src="https://github.com/user-attachments/assets/b08b8ffa-2a0f-48fa-b229-39065c2e48c7" />

---

## 6. Adaptive Query Rewriting

If retrieved context is insufficient, the evaluator LLM rewrites the query.

Example:

```text
User: what does it return?
Rewritten: register function return value
```

**Benefit:** Better retrieval for ambiguous follow-up questions.

---

## 7. Multi-Hop Retrieval

If the main chunk is retrieved but its dependencies are needed, the LLM can request dependency implementations.

Example:

```json
{
  "symbol": "handle_resume_upload",
  "dependencies": [
    "extract_text_from_pdf",
    "clean_extracted_text"
  ]
}
```

The pipeline fetches those dependency chunks and sends them back to the LLM.

**Benefit:** Answers can include deeper implementation details across multiple functions.

---

## 8. Reranking Pipeline

Retrieved FAISS results are reranked using a reranker model before final answering.

This improves:

* Precision
* Context quality
* Top-k relevance

---

## 9. Short-Term Conversational Memory

The system maintains recent conversation history inside the prompt.

Example:

```text
User: is register implemented?
User: what does it return?
```

The pipeline resolves references like:

*"it"
*"that function"
*"this class"

using previous chat context.

**Benefit** : Enables natural multi-turn conversations about the codebase.

---

## 10. Context provided to LLM

The LLM is given :

* Previous conversations
* User query
* Retrieved chunks with one liner explanation of of dependencies

```text
Chunk 1: (Retrieve a conversation and return serialized response)
    def get(self, request, conv_id: int):
        '''Retrieve a specific conversation.'''
        conv = self._get_conv(request, conv_id)
        return Response(ConversationSerializer(conv).data)


Dependencies:
ConversationSerializer: DRF serializer for Conversation model
_get_conv: Fetch conversation object for requesting user
```

**Benefit** : Allows LLM to understand user's approach and get idea about dependencies used in the chunks.

---

## 11. Streaming Responses (SSE)

Answers are streamed token-by-token to the frontend using Server-Sent Events.

**Benefit:** Faster perceived response time and better UX.

---

## 12. Logging & Debugging

Logs include:

* Retrieval summaries
* Reranking output
* Dependency requests
* Query rewrites
* Final state summary
* Token usage
* Indexing diagnostics

---

# Tech Stack

## Backend

* **Python 3.12**
* **Django**
* **Django REST Framework**

## RAG / LLM

* **LangChain**
* **Groq API** (fast explanation generation)

## Vector Database

* **FAISS**

## Embeddings

* **Custom ONNX embedding model**
* **onnxruntime**
* **tokenizers**

## Parsing / Processing

* **Python AST**
* **PyMuPDF** (PDF support)

## Storage

* **PostgreSQL** (metadata & chunk tracking)

---

# RAG Pipeline Workflow

<img width="1536" height="1024" alt="ChatGPT Image May 12, 2026, 01_58_13 PM" src="https://github.com/user-attachments/assets/50a72284-960a-4efe-9ead-0aa657132dff" />

---

# RAG Evaluation Results

## Retrieval Quality Evaluation

```json
"summary": {
    "total_queries": 90,
    "recall": 0.8767,
    "mrr": 0.8165,
    "hit_rate": 0.9889
  }
```
**Results in evaluation/results_pipeline.json**

## Generation Quality Evaluation (Out of 5)

```json
"summary": {
    "total_queries": 90,
    "avg_correctness": 4.52,
    "avg_grounding": 4.35,
    "avg_completeness": 4.58,
    "hallucination_rate": 0.23
  }
```
**Results in evaluation/results_generation.json**

## Failure Handling

* Query rewrite fallback
* Force-answer after max retries
* Graceful handling of missing chunks

---

## Screenshot of website

<img width="2000" height="1200" alt="{001DF7AA-D7C3-4370-9689-DECCF82C8310}" src="https://github.com/user-attachments/assets/0beb3aa1-c4be-4915-8bc9-45227a5afd63" />

---

# Key Highlights

✅ Code-aware semantic search 
✅ Explanation-based embeddings
✅ Dependency graph generation
✅ Adaptive query rewriting
✅ Multi-hop dependency retrieval
✅ Incremental re-indexing
✅ FAISS vector search
✅ Streaming answers
✅ Debug logging

---

# Future Improvements

* LangGraph agent workflow
* Cross-file reasoning
* Better reranking models
* Support for multiple programming languages
* UI for dependency graph visualization

---

## Deployment link

https://codemind-2li4.onrender.com
