  graph_builder = StateGraph(State).add_sequence([retrieve_wrapper, generate_wrapper])
    graph_builder.add_edge(START, "retrieve_wrapper")
    graph = graph_builder.compile()