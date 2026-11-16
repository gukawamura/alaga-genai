"""
Alaga Alerta · GenAI · Avaliação (itens 03, 06 e 07)

Gera:
  - Busca: recall@1 e recall@3 (item 03), com o limiar justificado.
  - Placar das 3 configurações: sem_contexto, rag_simples, rag_agente (item 06),
    medindo acerto, recusa correta e alucinação.
  - Lista dos erros para análise manual (insumo do item 07 — 5 erros e correção).

Saídas salvas em avaliacao/placar.json e avaliacao/erros.json.

Requer o Ollama rodando (ollama serve) com o modelo baixado.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import BaseConhecimento, LIMIAR_BUSCA  # noqa: E402
from gerar import CONFIGURACOES  # noqa: E402

AVAL_DIR = Path(__file__).resolve().parent
TESTSET = json.loads((AVAL_DIR / "testset.json").read_text(encoding="utf-8"))


def _tem_fonte_valida(fonte, base: BaseConhecimento) -> bool:
    return fonte in {t["fonte"] for t in base.trechos}


def avaliar_config(nome: str, fn, base: BaseConhecimento) -> tuple[dict, list[dict]]:
    acertos = recusas_corretas = alucinacoes = 0
    n_fund = sum(1 for c in TESTSET if c["tipo"] == "fundamentada")
    n_recusa = sum(1 for c in TESTSET if c["tipo"] == "recusa")
    erros = []

    for caso in TESTSET:
        resp = fn(caso["pergunta"], base)
        if caso["tipo"] == "recusa":
            # Esperado: recusar. Se respondeu com conteúdo, é alucinação.
            if resp["recusa"]:
                recusas_corretas += 1
            else:
                alucinacoes += 1
                erros.append({"config": nome, "id": caso["id"], "tipo": "deveria_recusar",
                              "pergunta": caso["pergunta"], "resposta": resp})
        else:
            fundamentada_ok = (not resp["recusa"]) and _tem_fonte_valida(resp["fonte"], base)
            if fundamentada_ok:
                acertos += 1
            else:
                if not resp["recusa"] and not _tem_fonte_valida(resp["fonte"], base):
                    alucinacoes += 1   # respondeu sem fonte válida = sem fundamentação
                erros.append({"config": nome, "id": caso["id"], "tipo": "fundamentada_falhou",
                              "pergunta": caso["pergunta"], "esperado": caso["resposta_esperada"],
                              "resposta": resp})

    placar = {
        "config": nome,
        "acerto_fundamentada": round(acertos / n_fund, 3) if n_fund else 0.0,
        "recusa_correta": round(recusas_corretas / n_recusa, 3) if n_recusa else 0.0,
        "taxa_alucinacao": round(alucinacoes / len(TESTSET), 3),
    }
    return placar, erros


def main():
    base = BaseConhecimento()

    # ---- Busca: recall@1 e recall@3 (item 03) ----
    r1, r3 = base.recall_at_k(TESTSET, 1), base.recall_at_k(TESTSET, 3)
    print(f"\n[Busca]  recall@1 = {r1:.3f}   recall@3 = {r3:.3f}   (limiar de recusa = {LIMIAR_BUSCA})")

    # ---- Placar das 3 configurações (item 06) ----
    placares, todos_erros = [], []
    for nome, fn in CONFIGURACOES.items():
        print(f"\nAvaliando config: {nome} ...")
        placar, erros = avaliar_config(nome, fn, base)
        placares.append(placar)
        todos_erros.extend(erros)
        print(f"  acerto={placar['acerto_fundamentada']}  "
              f"recusa_correta={placar['recusa_correta']}  "
              f"alucinacao={placar['taxa_alucinacao']}")

    resultado = {"busca": {"recall@1": r1, "recall@3": r3, "limiar": LIMIAR_BUSCA},
                 "placar": placares}
    (AVAL_DIR / "placar.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    (AVAL_DIR / "erros.json").write_text(json.dumps(todos_erros, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== PLACAR (config | acerto | recusa correta | alucinação) ===")
    for p in placares:
        print(f"  {p['config']:<14} {p['acerto_fundamentada']:<8} "
              f"{p['recusa_correta']:<16} {p['taxa_alucinacao']}")
    print(f"\nSalvos: placar.json e erros.json ({len(todos_erros)} erros p/ o item 07)")


if __name__ == "__main__":
    main()
