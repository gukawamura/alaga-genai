# Alaga Alerta — Componente **Inteligência** (Generative AI for Engineering)

Assistente que responde o cidadão sobre risco e segurança em alagamentos
("o que faço se meu carro ficar preso na água?", "essa rua costuma alagar?"),
com base em manuais da Defesa Civil / boletins do CGE e nos dados da camada Ouro,
**citando a fonte** e dizendo **"não consta"** quando não sabe.

Prova central da disciplina: uma LLM local **acerta mais e alucina menos** com
RAG + agente do que o mesmo modelo sem contexto.

---

## 1. Integrantes

- Felipe Capriotti — RM98460
- Manoella Herrerias Waideman — RM98906
- Victor Hugo Andrade — RM550996
- Gustavo Kawamura — RM99679

## 2. Ficha Técnica — o que a IA resolve

| Campo | Valor |
|---|---|
| Tarefa da IA | QA fundamentado (RAG) sobre segurança em alagamentos |
| Entrada | Pergunta do cidadão (texto), via chat do MVP |
| Saída | JSON `{resposta, fonte, recusa}` |
| Modelo | Llama 3.2 3B local (Ollama), configurável via `OLLAMA_MODEL`; embeddings `multilingual-e5-small` |
| Base | Manuais Defesa Civil/Bombeiros/Saúde, CGE e camada Ouro (Big Data) |
| Integração | `POST /assistente` consumido pelo chat do MVP |
| LGPD | Nenhum dado pessoal é enviado ao modelo |

## 3. Diagrama do pipeline de IA e dos documentos

```
Documentos (Defesa Civil, CGE, Saúde, Ouro)
        │  (chunking manual + fonte/licença)
        ▼
 base/conhecimento.json ──► embeddings e5 (normalizados)
                                   │
Pergunta do cidadão ──► [agente] reformula ──► busca cosseno (top-k)
                                   │
                    score < limiar(0.84)? ── sim ─► "não consta" (recusa)
                                   │ não
                                   ▼
                    Llama 3.1 8B (JSON, cita a fonte) ──► autoverificação da fonte
                                   │
                                   ▼
                        POST /assistente ──► chat do MVP
```

## 4. Tabela da base de conhecimento

| Fonte | Formato | Licença | Trechos |
|---|---|---|---|
| Defesa Civil (orientações/prevenção/evacuação) | texto curado | Conteúdo público (Governo) | 8 |
| Corpo de Bombeiros | texto curado | Conteúdo público (Governo) | 1 |
| Secretaria de Saúde (pós-enchente, leptospirose) | texto curado | Conteúdo público (Governo) | 2 |
| CGE São Paulo (estados de atenção) | texto curado | Conteúdo público (Governo) | 1 |
| Alaga Alerta — camada Ouro (Big Data) | texto curado | Projeto | 2 |
| **Total** | | | **14** |

## 5. Performance nas 3 configurações

Execução real com `python avaliacao/avaliar.py` (modelo `llama3.2:3b`, 14 casos: 10
fundamentadas + 4 de recusa). Saídas em `avaliacao/placar.json` e `avaliacao/erros.json`.

| Configuração | Acerto (fundamentada) | Recusa correta | Taxa de alucinação |
|---|---|---|---|
| sem_contexto | 0.00 | 0.00 | **1.00** |
| rag_simples  | 0.40 | 1.00 | 0.43 |
| rag_agente   | **1.00** | **1.00** | **0.00** |

Busca: **recall@1 = 1.00**, **recall@3 = 1.00**. Limiar de recusa = **0.84**,
escolhido porque nos testes os trechos relevantes ficam ≥ 0.84 (ex.: 0.886 / 0.884 / 0.857
para uma pergunta no domínio) e perguntas fora de domínio caem abaixo — o que separa
"responder fundamentado" de "não consta".

> Leitura dos números: `sem_contexto` responde a tudo e **alucina 100%** — nunca cita fonte
> válida e ainda responde perguntas fora de domínio (deveria recusar). `rag_simples` já recusa
> corretamente as 4 fora de domínio (recusa 1.00) e reduz a alucinação para 0.43, mas perde
> acerto porque **malformata a citação** (copia `[c03] (fonte: …)` ou só o id `c10` em vez da
> fonte limpa). `rag_agente` chega a **acerto 1.00 e alucinação 0.00**: a reformulação melhora
> o recall, o limiar barra o fora de domínio e a **autoverificação ancora a `fonte`** no trecho
> recuperado, eliminando a citação inválida. Prova a tese: a mesma LLM local **acerta mais e
> alucina menos** com RAG + agente.

## 6. Os 5 erros (causa e correção)

Casos reais extraídos de `avaliacao/erros.json` (20 erros registrados). Selecionamos 5 que
cobrem os padrões distintos:

| # | Caso | Erro observado | Causa provável | Correção proposta |
|---|---|---|---|---|
| 1 | `sem_contexto` t03 — "Qual o telefone da Defesa Civil?" | Respondeu **188** (o correto é **199**), sem fonte | Sem base, o modelo "chuta" um número plausível | RAG traz o telefone correto da base e cita a fonte; limiar recusa quando não há suporte |
| 2 | `sem_contexto` t10 — "Quais bairros o Alaga Alerta monitora?" | Inventou bairros (Jardins, Moema, Guarulhos) que **não** são monitorados | Modelo não conhece o escopo do projeto e alucina a lista | RAG recupera os bairros reais da base/camada Ouro |
| 3 | `sem_contexto` t11 — "Previsão do tempo para o Rio de Janeiro amanhã?" | Respondeu em vez de **recusar** (pergunta fora de domínio) | Baseline não tem mecanismo de recusa | Limiar de busca < 0.84 → "não consta" (no `rag_agente` a recusa correta é 1.00) |
| 4 | `rag_simples` t01 — "Carro preso na água?" | Conteúdo **correto**, mas citou `[c03] (fonte: Corpo de Bombeiros…)` — formato inválido → contabilizado como não-fundamentado | O modelo copia o rótulo do contexto em vez da fonte limpa | Passo de **autoverificação** do agente ancora a `fonte` no trecho recuperado |
| 5 | `rag_simples` t04 — "O que é risco crítico?" | Citou só o id `c10` como fonte (inválida) e deu um limiar impreciso | Sem validação da citação após a geração | Autoverificação do agente substitui pela fonte real do top-1 recuperado |

> Observação: no `rag_simples` vários "erros" são de **citação malformatada**, não de conteúdo —
> por isso o `rag_agente`, que valida e ancora a fonte, salta para acerto 1.00.

## 7. Conclusão técnica e links

- Repositório (este): https://github.com/gukawamura/alaga-genai
- Repositório Big Data (Ouro que alimenta a base): https://github.com/gukawamura/alaga-big-data
- Vídeo: https://youtu.be/qi57GOOAfyo

---

## Como reproduzir

```bash
# 1. Ollama (LLM local)
#    instale em https://ollama.com e baixe o modelo:
ollama pull llama3.1:8b        # máquina modesta: ollama pull llama3.2:3b
ollama serve                   # deixa o servidor em http://localhost:11434

# 2. Dependências Python
pip install -r requirements.txt

# 3. Testar a busca
python src/rag.py

# 4. Avaliação completa (recall + placar das 3 configs)
python avaliacao/avaliar.py

# 5. Subir a API consumida pelo MVP
uvicorn api.main:app --reload --port 8001
#    POST http://localhost:8001/assistente  body: {"pergunta": "posso atravessar a rua alagada a pé?"}
```

Para trocar o modelo sem mexer no código (Windows PowerShell):
```powershell
$env:OLLAMA_MODEL = "llama3.2:3b"
```

## Estrutura
```
genai/
├─ base/conhecimento.json        # 14 trechos com fonte e licença (item 02)
├─ src/rag.py                     # embeddings + busca + recall@k (item 03)
├─ src/gerar.py                   # 3 configs; JSON+citação+recusa (itens 04,05,06)
├─ avaliacao/testset.json         # casos com gabarito, incl. "não consta" (item 01)
├─ avaliacao/avaliar.py           # placar + erros (itens 06,07)
├─ api/main.py                    # POST /assistente (integração MVP)
├─ notebooks/pipeline_rag.ipynb   # execução com saídas salvas
└─ requirements.txt  .gitignore  README.md
```
