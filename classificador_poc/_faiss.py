import faiss
import numpy as np

from classificador_poc._bedrock import build_embedding

def construir_indices_e_referencias(json_ncms):
    indices_por_nivel = {}
    referencias_por_nivel = {}

    def processar_nivel(filhos, nivel):
        textos = [f["Descricao"] for f in filhos]
        vetores = np.array([build_embedding(texto) for texto in textos])

        index = faiss.IndexFlatL2(vetores.shape[1])
        index.add(vetores)

        indices_por_nivel[nivel] = index
        referencias_por_nivel[nivel] = filhos

        for filho in filhos:
            if filho.get("Filhos"):
                processar_nivel(filho["Filhos"], nivel + 1)

    processar_nivel(json_ncms, nivel=0)

    return indices_por_nivel, referencias_por_nivel
