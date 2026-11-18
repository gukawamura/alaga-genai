"""
Alaga Alerta · GenAI · Geração nas 3 configurações (itens 04, 05, 06)

  1. sem_contexto  — LLM responde direto, sem base (baseline para medir alucinação)
  2. rag_simples   — recupera top-k e responde fundamentado, citando a fonte ou recusando
  3. rag_agente    — ESPECIALIZAÇÃO (item 05): reformula a pergunta, decide buscar,
                     recusa por limiar e faz autoverificação da citação

Todas retornam JSON válido: {"resposta": str, "fonte": str|null, "recusa": bool}  (item 04)

Modelo via variável de ambiente OLLAMA_MODEL (default llama3.2:3b).
"""
from __future__ import annotations

import json
import os

import requests

from rag import BaseConhecimento, LIMIAR_BUSCA

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

RECUSA_PADRAO = {"resposta": "Não consta na minha base de conhecimento.", "fonte": None, "recusa": True}


def _chamar_ollama(prompt: str, system: str = "", temperatura: float = 0.0) -> dict:
    """Chama o Ollama forçando saída JSON e devolve o dict já parseado."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "format": "json",       # Ollama garante JSON sintaticamente válido
            "stream": False,
            "options": {"temperature": temperatura},
        },
        timeout=180,
    )
    r.raise_for_status()
    try:
        dados = json.loads(r.json()["response"])
        return dados if isinstance(dados, dict) else {}
    except (ValueError, KeyError, TypeError):
        return {}


def _normalizar(d: dict) -> dict:
    resposta, fonte = d.get("resposta"), d.get("fonte")
    if d.get("recusa") is not False or not isinstance(resposta, str) or not resposta.strip():
        return dict(RECUSA_PADRAO)
    return {"resposta": resposta.strip(), "fonte": fonte.strip() if isinstance(fonte, str) and fonte.strip() else None, "recusa": False}


SYSTEM_ASSISTENTE = (
    "Você é o assistente do Alaga Alerta, que orienta cidadãos sobre risco e segurança "
    "em alagamentos em São Paulo. Responda em português, de forma simples e direta. "
    "Responda SEMPRE em JSON com as chaves: resposta (string), fonte (string ou null) e "
    "recusa (booleano)."
)


# --------------------------------------------------------------------------- #
# 1) Sem contexto
# --------------------------------------------------------------------------- #
def sem_contexto(pergunta: str) -> dict:
    prompt = (
        f"Pergunta do cidadão: {pergunta}\n"
        "Responda apenas com base no que você sabe. Preencha 'fonte' com null."
    )
    return _normalizar(_chamar_ollama(prompt, SYSTEM_ASSISTENTE))


# --------------------------------------------------------------------------- #
# 2) RAG simples
# --------------------------------------------------------------------------- #
def _montar_contexto(trechos: list[dict]) -> str:
    return "\n".join(f"[{t['id']}] (fonte: {t['fonte']}) {t['texto']}" for t in trechos)


def rag_simples(pergunta: str, base: BaseConhecimento, k: int = 3) -> dict:
    trechos = base.buscar(pergunta, k=k)
    contexto = _montar_contexto(trechos)
    prompt = (
        f"Contexto (trechos da base de conhecimento):\n{contexto}\n\n"
        f"Pergunta do cidadão: {pergunta}\n\n"
        "Responda USANDO SOMENTE o contexto acima. Em 'fonte', cite a fonte do trecho usado. "
        "Se a resposta NÃO estiver no contexto, defina recusa=true, resposta='Não consta na "
        "minha base de conhecimento.' e fonte=null. Nunca invente informação."
    )
    return _normalizar(_chamar_ollama(prompt, SYSTEM_ASSISTENTE))


# --------------------------------------------------------------------------- #
# 3) RAG com agente (especialização — item 05)
# --------------------------------------------------------------------------- #
def _reformular(pergunta: str) -> str:
    """Reescreve a pergunta para melhorar a busca (passo agêntico)."""
    d = _chamar_ollama(
        f"Reescreva a pergunta a seguir de forma clara e objetiva para busca em uma base "
        f"sobre alagamentos, mantendo o sentido. Pergunta: {pergunta}\n"
        'Responda em JSON: {"reformulada": "..."}',
    )
    consulta = d.get("reformulada")
    return consulta.strip() if isinstance(consulta, str) and consulta.strip() else pergunta


def rag_agente(pergunta: str, base: BaseConhecimento, k: int = 3) -> dict:
    # Passo 1: reformula para melhorar o recall
    consulta = _reformular(pergunta)
    trechos = base.buscar(consulta, k=k)

    # Passo 2: decisão de recusa por limiar (justificado na avaliação)
    melhor = trechos[0]["score"] if trechos else 0.0
    if melhor < LIMIAR_BUSCA:
        return dict(RECUSA_PADRAO)

    # Passo 3: geração fundamentada
    contexto = _montar_contexto(trechos)
    prompt = (
        f"Contexto:\n{contexto}\n\n"
        f"Pergunta do cidadão: {pergunta}\n\n"
        "Responda USANDO SOMENTE o contexto. Em 'fonte', escreva somente o ID do trecho que "
        "sustenta sua resposta, por exemplo c01, sem colchetes. "
        "Se o contexto não responder, recuse (recusa=true, fonte=null)."
    )
    resp = _normalizar(_chamar_ollama(prompt, SYSTEM_ASSISTENTE))

    # Passo 4: valida a referência escolhida pelo modelo antes de citar o trecho.
    if resp["recusa"]:
        return resp
    fontes = {t["id"]: t["fonte"] for t in trechos}
    fonte = fontes.get((resp["fonte"] or "").strip("[]").lower())
    if fonte is None:
        return dict(RECUSA_PADRAO)
    resp["fonte"] = fonte
    return resp


CONFIGURACOES = {
    "sem_contexto": lambda p, base: sem_contexto(p),
    "rag_simples": lambda p, base: rag_simples(p, base),
    "rag_agente": lambda p, base: rag_agente(p, base),
}


if __name__ == "__main__":
    base = BaseConhecimento()
    for nome, fn in CONFIGURACOES.items():
        print(f"\n=== {nome} ===")
        print(json.dumps(fn("o que faço se meu carro ficar preso na água?", base), ensure_ascii=False, indent=2))
