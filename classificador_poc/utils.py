import re
import json
from typing import List, Dict

_FOLDER =  "docs"
_FILE_TABELA_NCM_VIGENTE = "docs/Tabela_NCM_Vigente_20250415.json"
_FILE_TABELA_NCM_HIERARQUICO = "docs/Tabela_NCM_Vigente_20250415_hierarquico.json"
_FILE_TABELA_NCM_CONCATENADA = "docs/Tabela_NCM_Vigente_20250415_concatenada.json"

def clean_text(texto):
    texto = re.sub(r'<[^>]+>', '', texto)   # Remove tags HTML tipo <i>, <b>, etc
    texto = texto.replace("-", " ")         # Remove hífens e dois pontos
    texto = texto.replace(":", " ")
    texto = re.sub(r'\s+', ' ', texto)      # Remove espaços duplicados
    return texto.strip()

def build_json_ncm_hierarchical(ncms: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Constrói uma árvore hierárquica de NCMs corrigida com base na estrutura dos códigos.
    Cada nível é pai de códigos mais longos que começam com seu prefixo imediato.
    """
    def normalizar_codigo(codigo: str) -> str:
        return codigo.replace(".", "")

    # Ordena do código mais curto para o mais longo
    ncms_ordenados = sorted(ncms, key=lambda x: len(normalizar_codigo(x["Codigo"])))
    index = {normalizar_codigo(item["Codigo"]): item for item in ncms_ordenados}

    for item in index.values():
        item["Filhos"] = []

    for codigo, item in index.items():
        codigo_normalizado = normalizar_codigo(codigo)
        if len(codigo_normalizado) == 2:
            continue  # capítulo (raiz)
        # Tenta encontrar o pai mais próximo (prefixo mais curto existente)
        for i in range(len(codigo_normalizado) - 1, 1, -1):
            prefixo_pai = codigo_normalizado[:i]
            if prefixo_pai in index:
                index[prefixo_pai]["Filhos"].append(item)
                break

    # Coleta apenas os capítulos (nível raiz)
    raiz = [item for codigo, item in index.items() if len(codigo) == 2]
    return {"NCMs": raiz}

def build_json_ncm_concatenated(json_ncms: List[Dict]) -> Dict[str, str]:
    resultado = {}

    def percorrer(no, contexto_acumulado):
        novo_contexto = contexto_acumulado + [clean_text(no["Descricao"])]

        if not no.get("Filhos"):  # se não tem filhos, é uma folha
            codigo_ncm = no["Codigo"]
            texto_concatenado = ' '.join(novo_contexto)
            resultado[codigo_ncm] = texto_concatenado
        else:
            for filho in no["Filhos"]:
                percorrer(filho, novo_contexto)

    for ncm in json_ncms['NCMs']:
        percorrer(ncm, [])

    return resultado

if __name__ == "__main__":

    with open(_FILE_TABELA_NCM_HIERARQUICO, "r", encoding="utf-8") as jsonFile:
        json_ncm_hierarquico = json.load(jsonFile)
    
    json_ncm_concatenado = build_json_ncm_concatenated(json_ncm_hierarquico)

    with open(_FILE_TABELA_NCM_CONCATENADA, "w", encoding="utf-8") as f:
        json.dump(json_ncm_concatenado, f, ensure_ascii=False, indent=2)