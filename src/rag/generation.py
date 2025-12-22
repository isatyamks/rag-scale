from __future__ import annotations

import logging
from typing import Dict, Any


def generate(state: Dict[str, Any], llm, prompt_unused=None) -> Dict[str, str]:
    try:
        from langchain_core.prompts import ChatPromptTemplate

        context_text = "\n\n".join(doc.page_content for doc in state.get("context", []))
        
        system_template = """You are a RAG assistant that answers questions STRICTLY based on the provided context.

Context:
{context}

CRITICAL RULES:
1. Answer ONLY using information from the context above
2. DO NOT use your own knowledge or training data
3. If the answer is not in the context, respond EXACTLY: "I cannot find the answer in the provided documents."
4. Keep your answer brief and concise (3-4 sentences maximum)
5. Do NOT add extra information beyond what's in the context

Question: {question}
Answer:"""

        prompt = ChatPromptTemplate.from_template(system_template)
        
        chain = prompt | llm
        response = chain.invoke({"question": state.get("question", ""), "context": context_text})
        
        return {"answer": response.content}
    except Exception as e:
        logging.error(f"Error in generation: {e}")
        return {"answer": f"Failed to generate response. Error: {e}"}
