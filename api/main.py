"""
Alaga Alerta · GenAI · API do assistente

Expõe o modelo ao MVP da equipe. O chat do MVP chama POST /assistente e recebe
uma resposta fundamentada em JSON, com a fonte citada ou "não consta".

Rodar:  uvicorn api.main:app --reload --port 8001   (a partir da pasta genai/)
O MVP encaminha somente a pergunta após filtrar padrões de CPF, e-mail e telefone.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests import RequestException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import BaseConhecimento  # noqa: E402
from gerar import rag_agente  # noqa: E402

app = FastAPI(title="Alaga Alerta — Assistente (GenAI)", version="1.0.0")

# Carrega a base e os embeddings uma vez, na subida da API.
base = BaseConhecimento()


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=1, max_length=500)


class Resposta(BaseModel):
    resposta: str
    fonte: str | None
    recusa: bool


@app.get("/health")
def health():
    return {"status": "ok", "trechos": len(base.trechos)}


@app.post("/assistente", response_model=Resposta)
def assistente(entrada: Pergunta):
    """Recebe a pergunta do cidadão e devolve resposta fundamentada (RAG + agente)."""
    pergunta = entrada.pergunta.strip()
    if not pergunta:
        raise HTTPException(422, "Informe uma pergunta.")
    try:
        return rag_agente(pergunta, base)
    except RequestException:
        raise HTTPException(503, "Modelo local indisponível.") from None
