"""
Alaga Alerta · GenAI · Avaliação (itens 03, 06 e 07)

Gera:
  - Busca: recall@1 e recall@3 (item 03), com o limiar justificado.
  - Placar das 3 configurações: sem_contexto, rag_simples, rag_agente (item 06),
    medindo acerto de conteúdo e citação, recusa correta e divergências do gabarito.
  - Lista dos erros para análise manual (insumo do item 07 — 5 erros e correção).

Saídas salvas em avaliacao/placar.json e avaliacao/erros.json.

Requer o Ollama rodando (ollama serve) com o modelo baixado.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import BaseConhecimento, LIMIAR_BUSCA  # noqa: E402
from gerar import CONFIGURACOES, OLLAMA_MODEL  # noqa: E402

AVAL_DIR = Path(__file__).resolve().parent
TESTSET = json.loads((AVAL_DIR / "testset.json").read_text(encoding="utf-8"))


def _confere_conteudo(resposta: str, criterios: list[str]) -> bool:
    texto = unicodedata.normalize("NFKD", resposta.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return all(re.search(criterio, texto) for criterio in criterios)


def avaliar_config(nome: str, fn, base: BaseConhecimento) -> tuple[dict, list[dict]]:
    conteudos_corretos = acertos = recusas_corretas = respostas_divergentes = 0
    n_fund = sum(1 for c in TESTSET if c["tipo"] == "fundamentada")
    n_recusa = sum(1 for c in TESTSET if c["tipo"] == "recusa")
    resultados = []

    for caso in TESTSET:
        resp = fn(caso["pergunta"], base)
        tipo_erro = None
        if caso["tipo"] == "recusa":
            if resp["recusa"]:
                recusas_corretas += 1
            else:
                tipo_erro = "deveria_recusar"
                respostas_divergentes += 1
        elif resp["recusa"]:
            tipo_erro = "recusa_indevida"
        elif not _confere_conteudo(resp["resposta"], caso["criterios"]):
            tipo_erro = "conteudo_divergente_ou_incompleto"
            respostas_divergentes += 1
        else:
            conteudos_corretos += 1
            fontes_esperadas = {t["fonte"] for t in base.trechos if t["id"] in caso["gold_ids"]}
            if resp["fonte"] in fontes_esperadas:
                acertos += 1
            else:
                tipo_erro = "citacao_ausente_ou_incorreta"
        resultados.append({
            "config": nome, "id": caso["id"], "tipo_erro": tipo_erro,
            "pergunta": caso["pergunta"], "esperado": caso["resposta_esperada"], "resposta": resp,
        })

    placar = {
        "config": nome,
        "acerto_conteudo": round(conteudos_corretos / n_fund, 3),
        "acerto_fundamentada": round(acertos / n_fund, 3) if n_fund else 0.0,
        "recusa_correta": round(recusas_corretas / n_recusa, 3) if n_recusa else 0.0,
        "taxa_respostas_divergentes": round(respostas_divergentes / len(TESTSET), 3),
    }
    return placar, resultados


def main():
    base = BaseConhecimento()

    # ---- Busca: recall@1 e recall@3 (item 03) ----
    r1, r3 = base.recall_at_k(TESTSET, 1), base.recall_at_k(TESTSET, 3)
    print(f"\n[Busca]  recall@1 = {r1:.3f}   recall@3 = {r3:.3f}   (limiar de recusa = {LIMIAR_BUSCA})")

    # ---- Placar das 3 configurações (item 06) ----
    placares, todos_resultados = [], []
    for nome, fn in CONFIGURACOES.items():
        print(f"\nAvaliando config: {nome} ...")
        placar, resultados = avaliar_config(nome, fn, base)
        placares.append(placar)
        todos_resultados.extend(resultados)
        print(f"  conteudo={placar['acerto_conteudo']}  acerto_fundamentado={placar['acerto_fundamentada']}  "
              f"recusa_correta={placar['recusa_correta']}  "
              f"divergencias={placar['taxa_respostas_divergentes']}")

    resultado = {"modelo": OLLAMA_MODEL, "busca": {"recall@1": r1, "recall@3": r3, "limiar": LIMIAR_BUSCA},
                 "placar": placares, "respostas": todos_resultados}
    todos_erros = [r for r in todos_resultados if r["tipo_erro"]]
    (AVAL_DIR / "placar.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    (AVAL_DIR / "erros.json").write_text(json.dumps(todos_erros, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSalvos: placar.json e erros.json ({len(todos_erros)} erros p/ o item 07)")


if __name__ == "__main__":
    main()
