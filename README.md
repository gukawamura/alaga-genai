# Alaga Alerta — Componente **Inteligência** (Generative AI for Engineering)

Assistente que responde a perguntas sobre segurança em alagamentos
("o que faço se meu carro ficar preso na água?", "qual o telefone da Defesa Civil?"),
com base em orientações públicas, nos estados de atenção do CGE e em um recorte da camada Ouro histórica,
**citando a fonte** e dizendo **"não consta"** quando não sabe.

O projeto compara uma LLM local em três configurações usando o mesmo conjunto de perguntas.

---

## 1. Integrantes

- Felipe Capriotti — RM98460
- Manoella Herrerias Waideman — RM98906
- Victor Hugo Andrade — RM550996
- Gustavo Kawamura — RM99679

## 2. Ficha Técnica — o que a IA resolve

| Campo | Valor |
|---|---|
| Tarefa da IA | Perguntas e respostas fundamentadas (RAG) sobre segurança em alagamentos |
| Entrada | Pergunta do cidadão (texto), via chat do MVP |
| Saída | JSON `{resposta, fonte, recusa}` |
| Modelo | Llama 3.2 3B local (Ollama), configurável via `OLLAMA_MODEL`; embeddings `multilingual-e5-small` |
| Base | Orientações de Defesa Civil, Bombeiros e Saúde, estados de atenção do CGE e recorte histórico da Ouro |
| Integração | `POST /assistente` consumido pelo chat do MVP |
| LGPD | O MVP encaminha somente a pergunta e filtra padrões de CPF, e-mail e telefone. O filtro tem alcance limitado. Os testes usam perguntas sem dados pessoais. |

## 3. Diagrama do pipeline de IA e dos documentos

```
Documentos (Defesa Civil, CGE, Saúde, Ouro)
        │  (chunking manual + fonte/licença)
        ▼
 base/conhecimento.json ──► embeddings e5 (normalizados)
                                   │
Chat do MVP ──► POST /assistente na API central ──► filtro de dados pessoais
        │
        └──► API GenAI local ──► [agente] reformula ──► busca cosseno (top-3)
                                   │
                    score < limiar(0.84)? ── sim ─► "não consta" (recusa)
                                   │ não
                                   ▼
                    Llama 3.2 3B no Ollama ──► confere o ID de trecho citado
                                   │
                                   ▼
                        JSON ──► API central ──► chat do MVP
```

## 4. Tabela da base de conhecimento

| Fonte | Formato | Licença | Trechos |
|---|---|---|---|
| Defesa Civil (orientações/prevenção/evacuação) | resumo da equipe em JSON | Site público, licença específica não identificada | 7 |
| Corpo de Bombeiros | resumo da equipe em JSON | Site público, licença específica não identificada | 1 |
| Secretaria de Saúde (pós-enchente, leptospirose) | resumo da equipe em JSON | Site público, licença específica não identificada | 2 |
| CGE São Paulo (estados de atenção) | resumo da equipe em JSON | Site público, licença específica não identificada | 1 |
| Alaga Alerta — camada Ouro histórica e guia de risco | texto curado | Projeto | 3 |
| **Total** | | | **14** |

## 5. Performance nas 3 configurações

Executamos o mesmo modelo, `llama3.2:3b`, com temperatura 0 nas três
configurações. A especialização foi implementada como agente de RAG, mantendo os
pesos originais do modelo. São 14 perguntas, com 10 gabaritos de conteúdo e 4
casos em que se espera uma recusa. O teste, as 42 respostas e os erros estão
em `avaliacao/testset.json`, `avaliacao/placar.json` e `avaliacao/erros.json`.

- **Acerto de conteúdo** mede a presença dos conceitos obrigatórios do gabarito
  nas 10 perguntas fundamentadas. Os critérios de cada pergunta estão no JSON.
- **Acerto fundamentado** exige acerto de conteúdo e citação correspondente a
  um dos trechos esperados. O agente valida o ID do trecho escolhido pelo
  modelo antes de devolver a fonte.
- **Recusa correta** conta quantos dos 4 casos sem resposta foram recusados.
- **Respostas divergentes** contam, entre os 14 casos, as respostas com
  conteúdo divergente ou incompleto e as respostas a perguntas que deveriam
  ser recusadas. É um indicador de risco de alucinação. O teste por padrões
  pode perder paráfrases e afirmações inventadas que apareçam junto com a
  resposta correta. Erros só de citação e recusas indevidas são registrados
  separadamente em `erros.json`.

| Configuração | Acerto de conteúdo | Acerto fundamentado | Recusa correta | Respostas divergentes |
|---|---:|---:|---:|---:|
| Sem contexto | 3/10 | 0/10 | 0/4 | 11/14 |
| RAG simples | 9/10 | 3/10 | 4/4 | 1/14 |
| RAG com agente | 9/10 | 7/10 | 4/4 | 1/14 |

O RAG melhorou o conteúdo e a recusa em relação ao modelo sem contexto.
O agente melhorou a citação em relação ao RAG simples, com o mesmo acerto
de conteúdo e a mesma taxa de recusa nessa rodada. Ainda houve duas
citações de trechos que não sustentavam a resposta e uma resposta incompleta.
A inspeção também encontrou um erro factual no caso `t09` do modelo sem contexto,
mesmo tendo passado pelos padrões dos sintomas. Isso ilustra o limite da
métrica automática.

A busca sobre as perguntas originais obteve **recall@1 = 1,00** e
**recall@3 = 1,00**. A similaridade do primeiro trecho variou de **0,872 a
0,927** nas perguntas com gabarito e de **0,786 a 0,839** nas de recusa.
Escolhemos **0,84** com base nessa amostra. O agente aplica o limiar após
reformular a pergunta, e o efeito combinado aparece nas métricas de recusa.
O conjunto é pequeno e foi usado para calibrar o limiar.

## 6. Os 5 erros (causa e correção)

Exemplos da execução de 18/11/2026. As respostas completas estão em `avaliacao/erros.json`.

| Caso | Erro observado | Causa provável | Correção proposta |
|---|---|---|---|
| Sem contexto · `t01` | Sugeriu uma improvisação perigosa com um objeto entre o carro e a terra | Sem base, gerou instruções que não constam nas orientações | Recuperar o trecho oficial e validar a orientação antes de exibi-la |
| Sem contexto · `t03` | Indicou **188** para a Defesa Civil, cujo telefone na base é **199** | Produziu um número sem uma referência | Usar o trecho de contatos e validar números de emergência antes de responder |
| Sem contexto · `t05` | Deu a faixa de 200 a 300 mm por mês. O gabarito é de 10 até menos de 25 mm/h | O modelo desconhece a regra da camada Ouro histórica | Vincular limites e unidades ao trecho que descreve a regra |
| RAG simples · `t01` | Devolveu o texto da fonte com o rótulo `[c03] (fonte: …)` | Copiou a linha de contexto para o campo de fonte | Pedir só o ID do trecho e converter esse ID na fonte pública, como faz o agente |
| RAG com agente · `t05` | Acertou a faixa de chuva, mas citou orientações para grupos vulneráveis | O ID escolhido existia entre os trechos recuperados, mas não dava suporte aos números | Verificar se o trecho citado contém os valores e a unidade antes de retornar a resposta |


## 7. Conclusão técnica e links

A comparação sustenta o uso da base de conhecimento e da verificação de
citações. O principal ajuste seguinte é conferir se o trecho escolhido
sustenta cada afirmação, especialmente números e orientações de emergência.
O vídeo demonstra o projeto. O placar deste README corresponde à avaliação
revisada após a gravação.

- Repositório (este): https://github.com/gukawamura/alaga-genai
- Repositório Big Data (Ouro que alimenta a base): https://github.com/gukawamura/alaga-big-data
- MVP público: https://alaga-alerta-mvp.onrender.com/
- Vídeo: https://youtu.be/qi57GOOAfyo

---

## Como reproduzir

```bash
# 1. Ollama (LLM local)
#    instale em https://ollama.com e baixe o modelo:
ollama pull llama3.2:3b
ollama serve                   # se o Ollama ainda não estiver rodando

# 2. Dependências Python
pip install -r requirements.txt

# 3. Testar a busca
python src/rag.py

# 4. Avaliação completa (recall + placar das 3 configs)
python avaliacao/avaliar.py

# 5. Subir a API consumida pelo MVP
uvicorn api.main:app --host 127.0.0.1 --port 8001
#    POST http://localhost:8001/assistente  body: {"pergunta": "posso atravessar a rua alagada a pé?"}
```

No terminal do MVP, configure a URL e o tempo de espera antes de iniciar a API
central. Abra o chat na URL do MVP local. Em PowerShell:
```powershell
$env:ASSISTENTE_URL = "http://127.0.0.1:8001/assistente"
$env:TIMEOUT_SERVICOS = "75"
```

Uma resposta com `modo: "llm"` indica que a API central recebeu a saída do
serviço GenAI. No teste de integração de 18/11/2026, enviamos duas perguntas
à rota do MVP local. A pergunta sobre a Defesa Civil retornou **199** com
fonte e `consta: true`. A pergunta sobre o Campeonato Brasileiro retornou
**não consta**, sem fonte e com `consta: false`. As duas retornaram HTTP 200
e `modo: "llm"`.

O MVP público no Render atende com a base de contingência
(`modo: "local"`). A avaliação usa Ollama na máquina local. A base contém
um recorte estático dos bairros e limiares da Ouro histórica, com referência
ao código de origem. Previsão vigente e alertas são consultados nas telas do
MVP.

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
