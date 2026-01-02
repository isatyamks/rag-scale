from typing import List, Dict, Tuple

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.history: List[Tuple[str, str]] = []
        self.max_history = max_history

    def add_interaction(self, question: str, answer: str):
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        if not self.history:
            return ""
        
        formatted = []
        for q, a in self.history:
            formatted.append(f"Human: {q}")
            formatted.append(f"Assistant: {a}")
        
        return "\n".join(formatted)

    def clear(self):
        self.history = []
