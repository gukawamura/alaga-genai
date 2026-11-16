"""
Alaga Alerta · GenAI · Recuperação (RAG)

Busca semântica sobre a base de conhecimento usando embeddings locais
(sentence-transformers, modelo multilíngue e5-small) e similaridade por cosseno.
Expõe também o cálculo de recall@1 e recall@3 (item 03).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_BASE = BASE_DIR / "base" / "conhecimento.json"
MODELO_EMB = "intfloat/multilingual-e5-small"

# Limiar de corte da busca, justificado na avaliação (avaliacao/avaliar.py):
# nos testes, trechos relevantes ficam com cosseno >= 0.84 e perguntas fora do
# domínio ficam abaixo disso. Abaixo do limiar, o assistente responde "não consta".
LIMIAR_BUSCA = 0.84


class BaseConhecimento:
    def __init__(self, caminho: Path = CAMINHO_BASE, modelo: str = MODELO_EMB):
        self.trechos = json.loads(Path(caminho).read_text(encoding="utf-8"))
        self.encoder = SentenceTransformer(modelo)
        # e5 exige prefixos "passage:" nos documentos e "query:" na busca.
        textos = [f"passage: {t['texto']}" for t in self.trechos]
        self.matriz = self.encoder.encode(textos, normalize_embeddings=True)

    def buscar(self, pergunta: str, k: int = 3) -> list[dict]:
        q = self.encoder.encode([f"query: {pergunta}"], normalize_embeddings=True)[0]
        scores = self.matriz @ q  # vetores normalizados -> produto interno = cosseno
        ordem = np.argsort(scores)[::-1][:k]
        return [{**self.trechos[i], "score": float(scores[i])} for i in ordem]

    def recall_at_k(self, testset: list[dict], k: int) -> float:
        """Fração de perguntas fundamentadas cujo trecho-gabarito aparece no top-k."""
        casos = [c for c in testset if c["tipo"] == "fundamentada"]
        if not casos:
            return 0.0
        acertos = 0
        for caso in casos:
            ids = {t["id"] for t in self.buscar(caso["pergunta"], k=k)}
            if ids & set(caso["gold_ids"]):
                acertos += 1
        return acertos / len(casos)


if __name__ == "__main__":
    base = BaseConhecimento()
    for t in base.buscar("meu carro ficou preso na enchente, o que faço?"):
        print(f"{t['score']:.3f}  [{t['id']}] {t['fonte']}")
