from classificador_poc._langgraph import get_classification_app

app = get_classification_app()

input_state = {
    "descricao_usuario": "alto-falante de estante estéreo Sonos",
    "vetor_usuario": None,
    "nivel_atual": 0,
    "caminho": [],
    "terminado": False
}

resultado = app.invoke(input_state)
print(resultado["caminho"])