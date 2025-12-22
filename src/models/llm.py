

from langchain_ollama.chat_models import ChatOllama


def create_llm(
    model: str = "mistral:7b-instruct-q4_K_M",
    num_gpu: int = 999,
    temperature: float = 0,
    num_ctx: int = 2048,
    num_predict: int = 512
) -> ChatOllama:
    return ChatOllama(
        model=model,
        num_gpu=num_gpu,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict
    )
