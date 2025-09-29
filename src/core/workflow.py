from langgraph.graph import START, StateGraph
from src.core.state import State

def create_workflow(vector_store, llm):
    from src.retrieval.retriever import retrieve
    from src.generation.generator import generate
    
    def retrieve_wrapper(state: State):
        return retrieve(state, vector_store)
    
    def generate_wrapper(state: State):
        return generate(state, llm)
    
    graph_builder = StateGraph(State).add_sequence([retrieve_wrapper, generate_wrapper])
    graph_builder.add_edge(START, "retrieve_wrapper")
    graph = graph_builder.compile()
    
    return graph