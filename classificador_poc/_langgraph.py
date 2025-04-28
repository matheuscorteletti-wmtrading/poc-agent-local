from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Estado que será utilizado no Grafo
class ClassficiationState(TypedDict):
    descricao_usuario : str
    vetor_usuario: Optional[List[float]]
    nivel_atual: int
    caminho: List[dict] # lista de {"codigo": str, "descricao":str}
    terminando: bool

# TODO: porque optou por fazer lazy import?
# Função para gerar embedding
def build_embedding_node(state: ClassficiationState) -> ClassficiationState:
    from classificador_poc._bedrock import build_embedding
    vetor = build_embedding(state['descricao_usuario'])
    state["vetor_usuario"] = vetor
    return state


# # Função que decide qual filho (do json) seguir no fluxo
# def search_son_node(state: ClassficiationState) -> ClassficiationState:

#     nivel           = state["nivel_atual"]
#     vetor_usuario   = state["vetor_usuario"]

#     index           = indices_por_nivel[nivel]
#     referencias     = referencias_por_nivel[nivel]

#     D, I = index.search


def get_classification_app() -> CompiledStateGraph:

    graph = StateGraph(ClassficiationState)

    graph.add_node("build_embedding", build_embedding_node)
    # graph.add_node("search_son", search_son_node)

    graph.set_entry_point("build_embedding")
    graph.add_edge("build_embedding", "search_son")
    graph.add_conditional_edges(
        "search_son",
        condition=lambda state: END if state["terminado"] else "search_son"
    )

    return graph.compile()