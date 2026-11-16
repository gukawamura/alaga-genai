"""
Alaga Alerta · GenAI · API do assistente

Expõe o modelo ao MVP da equipe. O chat do MVP chama POST /assistente e recebe
uma resposta fundamentada em JSON, com a fonte citada ou "não consta".

Rodar:  uvicorn api.main:app --reload --port 8001   (a partir da pasta genai/)
LGPD:   nenhum dado pessoal é enviado ao modelo — só a pergunta do cidadão.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import BaseConhecimento  # noqa: E402
from gerar import rag_agente  # noqa: E402

app = FastAPI(title="Alaga Alerta — Assistente (GenAI)", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Carrega a base e os embeddings uma vez, na subida da API.
base = BaseConhecimento()


class Pergunta(BaseModel):
    pergunta: str


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
    return rag_agente(entrada.pergunta, base)
