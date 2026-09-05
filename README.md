# 🏥 HealthSearch — Motor de Busca Híbrido Médico

Motor de busca híbrido inteligente que combina a precisão léxica do **Okapi BM25** com a inteligência contextual de **Embeddings Semânticos Vetoriais** através de **Reciprocal Rank Fusion (RRF)** e **Cross-Encoder Re-Ranking**.

Projeto desenvolvido para a disciplina de **Tendências em Ciência da Computação (Recuperação de Informação / Processamento de Linguagem Natural)** — UNIPÊ.

---

## 👥 Equipe

- **Miguel** (Parte 1: Fundação, Ingestão, Stopwords PT, Regex e BM25)
- **Rafael** (Parte 2: Embeddings Semânticos, Fusão RRF e Cross-Encoder)
- **Vitor** (Parte 3: Interface Streamlit, Matriz Comparativa e Relatório Técnico)

- ** Com auxílio de IA Generativa **
---

## 🚀 Como Executar

### 1. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 2. Rodar a aplicação Streamlit
```bash
streamlit run healthsearch_app.py
```

### 3. Gerar / Atualizar o Relatório Técnico em PDF
```bash
python generate_report.py
```

---

## 📄 Entregáveis do Projeto
- `healthsearch_app.py`: Aplicação completa interativa em Streamlit.
- `Relatorio_Tecnico_HealthSearch.pdf`: Relatório Técnico estruturado em 2 páginas.
- `generate_report.py`: Script gerador automatizado do relatório em PDF.
- `requirements.txt`: Especificação das dependências.