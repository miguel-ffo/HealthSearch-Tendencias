"""
==============================================================================
HealthSearch — Motor de Busca Híbrido Médico (BM25 + Semântico Vetorial + RRF)
==============================================================================
Instituição: UNIPÊ — Centro Universitário de João Pessoa
Disciplina: Tendências em Ciência da Computação (Recuperação de Informação / PLN)
Professor: Me. Ricardo Roberto de Lima
Equipe:
 - Miguel (Parte 1: Ingestão, Corpus, Pré-processamento & BM25)
 - Rafael (Parte 2: Motor Semântico Vetorial, Fusão RRF & Cross-Encoder)
 - Vitor (Parte 3: Interface Streamlit, Matriz Comparativa & Dashboard)
==============================================================================
"""

from __future__ import annotations
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None
    util = None


@dataclass
class Document:
    """Representa um documento técnico no repositório de inteligência clínica."""

    id: str
    title: str
    content: str
    category: str = "Diretriz Clínica"


class MedicalCorpus:
    """Repositório oficial dos 6 documentos de diretrizes clínicas do laboratório."""

    OFFICIAL_DOCS = [
        Document(
            id="Doc 1",
            title="Protocolo Emergência ECG",
            content="Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos.",
            category="Emergência Cardiológica",
        ),
        Document(
            id="Doc 2",
            title="Guia de Farmacologia Cardíaca",
            content="O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio.",
            category="Farmacologia",
        ),
        Document(
            id="Doc 3",
            title="Diretriz de Hipertensão Arterial",
            content="A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI.",
            category="Terapia Intensiva",
        ),
        Document(
            id="Doc 4",
            title="Manual de AVC Isquêmico",
            content="O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas.",
            category="Neurologia de Emergência",
        ),
        Document(
            id="Doc 5",
            title="Protocolo de Reanimação RCR",
            content="Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul.",
            category="Suporte Avançado de Vida",
        ),
        Document(
            id="Doc 6",
            title="Procedimentos de UTI Geral",
            content="Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria.",
            category="Terapia Intensiva",
        ),
    ]

    def __init__(self, docs: Optional[List[Document]] = None):
        self.docs = docs or list(self.OFFICIAL_DOCS)
        self._doc_map = {d.id: d for d in self.docs}

    def get_documents(self) -> List[Document]:
        return self.docs

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        return self._doc_map.get(doc_id)


class TextPreprocessor:
    """Pipeline de processamento de linguagem natural focado em termos médicos."""

    PORTUGUESE_STOPWORDS = {
        "a",
        "à",
        "ao",
        "aos",
        "as",
        "às",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "é",
        "em",
        "entre",
        "era",
        "essa",
        "essas",
        "esse",
        "esses",
        "esta",
        "estas",
        "este",
        "estes",
        "eu",
        "foi",
        "foram",
        "fosse",
        "ha",
        "há",
        "isso",
        "isto",
        "já",
        "lhe",
        "lhes",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "meus",
        "minha",
        "minhas",
        "na",
        "nas",
        "não",
        "no",
        "nos",
        "nossa",
        "nossas",
        "nosso",
        "nossos",
        "num",
        "numa",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quando",
        "que",
        "quem",
        "se",
        "seja",
        "sem",
        "ser",
        "será",
        "seu",
        "seus",
        "só",
        "sua",
        "suas",
        "também",
        "te",
        "tem",
        "têm",
        "tinha",
        "tu",
        "tua",
        "tuas",
        "um",
        "uma",
        "você",
        "vocês",
        "ate",
        "até",
        "deles",
        "delas",
        "dele",
        "dela",
    }

    TOKEN_REGEX = re.compile(
        r"[a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]+(?:-[a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]+)*"
    )

    def clean_text(self, text: str) -> str:
        """Normaliza o texto convertendo para minúsculas e ajustando espaços."""
        if not text:
            return ""
        return text.strip().lower()

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokeniza preservando códigos compostos e removendo stopwords quando solicitado."""
        cleaned = self.clean_text(text)
        raw_tokens = self.TOKEN_REGEX.findall(cleaned)

        if not remove_stopwords:
            return raw_tokens

        return [
            t for t in raw_tokens if t not in self.PORTUGUESE_STOPWORDS and len(t) > 0
        ]


@dataclass
class SearchResult:
    """Estrutura padronizada de resultado individual de recuperação."""

    doc_id: str
    score: float
    rank: int
    title: str = ""
    content: str = ""
    category: str = ""
    bm25_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    cross_score: Optional[float] = None


class BM25SearchEngine:
    """Motor de recuperação léxica baseado no algoritmo Okapi BM25 com calibração dinâmica."""

    def __init__(
        self, corpus: MedicalCorpus, preprocessor: Optional[TextPreprocessor] = None
    ):
        self.corpus = corpus
        self.preprocessor = preprocessor or TextPreprocessor()
        self.docs = self.corpus.get_documents()
        self.tokenized_corpus = [
            self.preprocessor.tokenize(f"{d.title} {d.content}") for d in self.docs
        ]

    def search(
        self, query: str, k1: float = 1.2, b: float = 0.75
    ) -> List[SearchResult]:
        """Executa a busca léxica Okapi BM25 recalculando pontuações com k1 e b."""
        query_tokens = self.preprocessor.tokenize(query)

        if not query_tokens:
            return [
                SearchResult(
                    doc_id=d.id,
                    score=0.0,
                    rank=i + 1,
                    title=d.title,
                    content=d.content,
                    category=d.category,
                )
                for i, d in enumerate(self.docs)
            ]

        if BM25Okapi is not None:
            bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)
            raw_scores = bm25.get_scores(query_tokens)
        else:
            raw_scores = np.zeros(len(self.docs))
            for i, doc_toks in enumerate(self.tokenized_corpus):
                score = 0.0
                for qt in query_tokens:
                    if qt in doc_toks:
                        score += doc_toks.count(qt) * 1.5
                raw_scores[i] = score

        indexed_scores = []
        for doc_idx, raw_s in enumerate(raw_scores):
            clean_s = 0.0 if np.isnan(raw_s) else float(raw_s)
            indexed_scores.append((doc_idx, clean_s))

        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank_idx, (doc_idx, score) in enumerate(indexed_scores, start=1):
            doc = self.docs[doc_idx]
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    score=float(score),
                    rank=rank_idx,
                    title=doc.title,
                    content=doc.content,
                    category=doc.category,
                )
            )
        return results


class SemanticSearchEngine:
    """Motor de busca semântica vetorial com Embeddings Densos e Similaridade de Cosseno."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, corpus: MedicalCorpus, model_name: Optional[str] = None):
        self.corpus = corpus
        self.docs = self.corpus.get_documents()
        self.model_name = model_name or self.MODEL_NAME
        self.model = None
        self._doc_embeddings = None
        self._init_model()

    def _init_model(self):
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                try:
                    self.model = SentenceTransformer(self.FALLBACK_MODEL_NAME)
                except Exception:
                    self.model = None

        if self.model is not None:
            doc_texts = [f"{d.title}. {d.content}" for d in self.docs]
            self._doc_embeddings = self.model.encode(
                doc_texts, convert_to_tensor=True, normalize_embeddings=True
            )

    def search(self, query: str) -> List[SearchResult]:
        """Calcula a similaridade de cosseno entre a consulta e os documentos médicos."""
        if not query or not query.strip():
            return [
                SearchResult(
                    doc_id=d.id,
                    score=0.0,
                    rank=i + 1,
                    title=d.title,
                    content=d.content,
                    category=d.category,
                )
                for i, d in enumerate(self.docs)
            ]

        if (
            self.model is not None
            and self._doc_embeddings is not None
            and util is not None
        ):
            query_embedding = self.model.encode(
                query, convert_to_tensor=True, normalize_embeddings=True
            )
            cos_scores = util.cos_sim(query_embedding, self._doc_embeddings)[0]
            scores_np = cos_scores.cpu().numpy()
        else:
            scores_np = np.zeros(len(self.docs))
            q_words = set(query.lower().split())
            for i, d in enumerate(self.docs):
                d_words = set(f"{d.title} {d.content}".lower().split())
                overlap = len(q_words.intersection(d_words)) / max(len(q_words), 1)
                scores_np[i] = overlap

        scores_np = np.clip(scores_np, 0.0, 1.0)
        indexed_scores = list(enumerate(scores_np))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank_idx, (doc_idx, score) in enumerate(indexed_scores, start=1):
            doc = self.docs[doc_idx]
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    score=float(score),
                    rank=rank_idx,
                    title=doc.title,
                    content=doc.content,
                    category=doc.category,
                )
            )
        return results


class RRFFusionEngine:
    """Algoritmo de fusão Reciprocal Rank Fusion (RRF) para unificar rankings heterogêneos."""

    def fuse(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
        alpha: float = 0.5,
        k_rrf: int = 60,
    ) -> List[SearchResult]:
        """
        Calcula o score unificado RRF ponderado por alpha:
        Score_RRF(D) = alpha * [1 / (k_rrf + Rank_BM25)] + (1 - alpha) * [1 / (k_rrf + Rank_Semantico)]
        """
        bm25_map = {r.doc_id: r for r in bm25_results}
        sem_map = {r.doc_id: r for r in semantic_results}

        all_doc_ids = list(bm25_map.keys())
        doc_scores = []

        for doc_id in all_doc_ids:
            bm25_res = bm25_map.get(doc_id)
            sem_res = sem_map.get(doc_id)

            r_bm25 = bm25_res.rank if bm25_res else len(all_doc_ids) + 1
            r_sem = sem_res.rank if sem_res else len(all_doc_ids) + 1

            bm25_component = alpha * (1.0 / (k_rrf + r_bm25))
            sem_component = (1.0 - alpha) * (1.0 / (k_rrf + r_sem))
            rrf_score = bm25_component + sem_component

            ref_doc = bm25_res or sem_res
            doc_scores.append(
                {
                    "doc_id": doc_id,
                    "score": rrf_score,
                    "bm25_rank": r_bm25,
                    "semantic_rank": r_sem,
                    "title": ref_doc.title,
                    "content": ref_doc.content,
                    "category": ref_doc.category,
                }
            )

        if alpha >= 1.0:
            doc_scores.sort(key=lambda x: (x["bm25_rank"], -x["score"]))
        elif alpha <= 0.0:
            doc_scores.sort(key=lambda x: (x["semantic_rank"], -x["score"]))
        else:
            doc_scores.sort(key=lambda x: -x["score"])

        results = []
        for rank_idx, item in enumerate(doc_scores, start=1):
            results.append(
                SearchResult(
                    doc_id=item["doc_id"],
                    score=float(item["score"]),
                    rank=rank_idx,
                    title=item["title"],
                    content=item["content"],
                    category=item["category"],
                    bm25_rank=item["bm25_rank"],
                    semantic_rank=item["semantic_rank"],
                )
            )
        return results


class CrossEncoderReranker:
    """Re-ranking dos Top-K candidatos via Cross-Encoder (Desafio Bônus +0.3)."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self.model = None
        self._init_model()

    def _init_model(self):
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(self.model_name)
            except Exception:
                self.model = None

    def rerank(
        self, query: str, candidates: List[SearchResult], top_k: int = 3
    ) -> List[SearchResult]:
        """Aplica Cross-Encoder sobre os Top-K candidatos da busca híbrida."""
        if not candidates:
            return []

        top_candidates = [
            SearchResult(
                doc_id=c.doc_id,
                score=c.score,
                rank=c.rank,
                title=c.title,
                content=c.content,
                category=c.category,
                bm25_rank=c.bm25_rank,
                semantic_rank=c.semantic_rank,
                cross_score=c.cross_score,
            )
            for c in candidates[:top_k]
        ]
        rest_candidates = [
            SearchResult(
                doc_id=c.doc_id,
                score=c.score,
                rank=c.rank,
                title=c.title,
                content=c.content,
                category=c.category,
                bm25_rank=c.bm25_rank,
                semantic_rank=c.semantic_rank,
                cross_score=None,
            )
            for c in candidates[top_k:]
        ]

        pairs = [[query, f"{c.title}: {c.content}"] for c in top_candidates]

        if self.model is not None:
            try:
                raw_scores = self.model.predict(pairs)
                sigmoid_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
            except Exception:
                sigmoid_scores = np.array([c.score for c in top_candidates])
        else:
            sigmoid_scores = np.array([c.score for c in top_candidates])

        for c, score in zip(top_candidates, sigmoid_scores):
            c.cross_score = float(score)

        top_candidates.sort(
            key=lambda x: (x.cross_score if x.cross_score is not None else -1.0),
            reverse=True,
        )

        full_results = []
        for rank_idx, item in enumerate(top_candidates + rest_candidates, start=1):
            item.rank = rank_idx
            full_results.append(item)

        return full_results


def run_search_pipeline(
    query: str,
    k1: float = 1.2,
    b: float = 0.75,
    alpha: float = 0.5,
    enable_cross_encoder: bool = False,
    corpus: Optional[MedicalCorpus] = None,
    preprocessor: Optional[TextPreprocessor] = None,
    bm25_engine: Optional[BM25SearchEngine] = None,
    semantic_engine: Optional[SemanticSearchEngine] = None,
    rrf_engine: Optional[RRFFusionEngine] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> Dict[str, Any]:
    """Orquestra a execução unificada de todos os motores de recuperação e análise comparativa."""
    corpus = corpus or MedicalCorpus()
    preprocessor = preprocessor or TextPreprocessor()
    bm25_engine = bm25_engine or BM25SearchEngine(corpus, preprocessor)
    semantic_engine = semantic_engine or SemanticSearchEngine(corpus)
    rrf_engine = rrf_engine or RRFFusionEngine()
    reranker = reranker or CrossEncoderReranker()

    start_time = time.time()

    # 1. Recuperação Léxica
    bm25_res = bm25_engine.search(query, k1=k1, b=b)

    # 2. Recuperação Semântica
    semantic_res = semantic_engine.search(query)

    # 3. Fusão Híbrida RRF
    hybrid_res = rrf_engine.fuse(bm25_res, semantic_res, alpha=alpha, k_rrf=60)

    # 4. Re-ranking Opcional Cross-Encoder
    if enable_cross_encoder:
        hybrid_res = reranker.rerank(query, hybrid_res, top_k=3)

    latency_ms = (time.time() - start_time) * 1000.0

    # 5. Construção da Matriz Comparativa (DataFrame)
    bm25_dict = {r.doc_id: r for r in bm25_res}
    sem_dict = {r.doc_id: r for r in semantic_res}

    rows = []
    for h in hybrid_res:
        b_item = bm25_dict[h.doc_id]
        s_item = sem_dict[h.doc_id]
        rank_shift = abs(b_item.rank - s_item.rank)

        rows.append(
            {
                "ID": h.doc_id,
                "Título": h.title,
                "Categoria": h.category,
                "Rank BM25": b_item.rank,
                "Score BM25": round(b_item.score, 4),
                "Rank Semântico": s_item.rank,
                "Score Semântico": round(s_item.score, 4),
                "Rank Híbrido RRF": h.rank,
                "Score RRF": round(h.score, 6),
                "Score Cross-Encoder": (
                    round(h.cross_score, 4) if h.cross_score is not None else "-"
                ),
                "Rank Shift (|BM25 - Semântico|)": rank_shift,
                "Trecho": h.content,
            }
        )

    comparison_df = pd.DataFrame(rows)

    return {
        "query": query,
        "bm25_results": bm25_res,
        "semantic_results": semantic_res,
        "hybrid_results": hybrid_res,
        "comparison_df": comparison_df,
        "latency_ms": latency_ms,
        "alpha": alpha,
        "k1": k1,
        "b": b,
        "cross_encoder_enabled": enable_cross_encoder,
    }


def _render_document_card(result: SearchResult, mode: str = "bm25", query: str = ""):
    """Renderiza um card visual estético de documento no Streamlit."""
    if st is None:
        return

    rank_colors = {
        1: "#10B981",
        2: "#3B82F6",
        3: "#8B5CF6",
        4: "#64748B",
        5: "#94A3B8",
        6: "#CBD5E1",
    }
    badge_color = rank_colors.get(result.rank, "#64748B")

    with st.container():
        st.markdown(
            f"""
            <div style="
                border: 1px solid #E2E8F0;
                border-left: 5px solid {badge_color};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
                background-color: #FFFFFF;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="
                            background-color: {badge_color};
                            color: white;
                            padding: 3px 8px;
                            border-radius: 4px;
                            font-size: 0.85rem;
                            font-weight: bold;
                        ">Rank #{result.rank}</span>
                        <strong style="font-size: 1.05rem; margin-left: 8px; color: #0F172A;">{result.doc_id}: {result.title}</strong>
                    </div>
                    <span style="
                        background-color: #F1F5F9;
                        color: #475569;
                        padding: 3px 8px;
                        border-radius: 4px;
                        font-size: 0.8rem;
                    ">{result.category}</span>
                </div>
                <p style="color: #334155; font-size: 0.95rem; margin-bottom: 8px; line-height: 1.5;">
                    {result.content}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Exibição de métricas detalhadas abaixo do card
        if mode == "bm25":
            st.caption(f"🎯 **Score Okapi BM25**: `{result.score:.4f}`")
        elif mode == "semantic":
            st.caption(
                f"🧠 **Similaridade de Cosseno**: `{result.score:.4f}` ({result.score*100:.1f}%)"
            )
            st.progress(min(max(float(result.score), 0.0), 1.0))
        elif mode == "hybrid":
            cols = st.columns([1.5, 1, 1, 1.2])
            cols[0].caption(f"⚡ **Score RRF**: `{result.score:.6f}`")
            cols[1].caption(f"🔤 Rank BM25: `#{result.bm25_rank}`")
            cols[2].caption(f"🧠 Rank Sem.: `#{result.semantic_rank}`")
            if result.cross_score is not None:
                cols[3].caption(f"✨ Cross-Encoder: `{result.cross_score:.4f}`")


def main():
    """Ponto de entrada do aplicativo Streamlit."""
    if st is None:
        print("Streamlit não está instalado no ambiente.")
        return

    st.set_page_config(
        page_title="HealthSearch — Motor de Busca Híbrido Médico",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Cache de motores na sessão Streamlit para máxima performance
    if "corpus" not in st.session_state:
        st.session_state.corpus = MedicalCorpus()
        st.session_state.preprocessor = TextPreprocessor()
        st.session_state.bm25_engine = BM25SearchEngine(
            st.session_state.corpus, st.session_state.preprocessor
        )
        st.session_state.semantic_engine = SemanticSearchEngine(st.session_state.corpus)
        st.session_state.rrf_engine = RRFFusionEngine()
        st.session_state.reranker = CrossEncoderReranker()

    with st.sidebar:
        st.title("⚙️ Painel de Controle")
        st.markdown(
            "Calibre os parâmetros dos motores léxico e semântico em tempo real."
        )

        st.subheader("1. Motor Léxico (BM25)")
        k1 = st.slider(
            "Saturação de Frequência (k₁)",
            min_value=0.0,
            max_value=3.0,
            value=1.2,
            step=0.1,
            help="Controla a rapidez com que a frequência do termo satura no documento.",
        )
        b = st.slider(
            "Normalização por Comprimento (b)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Penaliza documentos mais longos para compensar o tamanho.",
        )

        st.divider()

        st.subheader("2. Fusão Híbrida (RRF)")
        alpha = st.slider(
            "Peso Balanceador (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="α = 1.0 (100% Léxico BM25) | α = 0.0 (100% Semântico Vetorial) | α = 0.5 (Equilibrado)",
        )

        st.divider()

        st.subheader("3. Desafio Bônus")
        enable_cross_encoder = st.checkbox(
            "Ativar Cross-Encoder Re-Ranking (+0.3)",
            value=False,
            help="Aplica modelo Cross-Encoder (ms-marco-MiniLM) sobre os Top-3 candidatos híbridos.",
        )

        st.divider()

        # Visualizador do Corpus na Sidebar
        with st.expander("📚 Ver Corpus Médico (6 Documentos)"):
            for doc in st.session_state.corpus.get_documents():
                st.markdown(f"**{doc.id}: {doc.title}**")
                st.caption(doc.content)

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 20px;">
            <b>HealthTech Solutions & UNIPÊ</b><br>
            Autores: Miguel, Rafael & Vitor<br>
            Prof. Me. Ricardo Roberto de Lima
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.title("🏥 HealthSearch — Motor de Busca Híbrido Médico")
    st.markdown(
        """
        Sistema inteligente de recuperação de diretrizes clínicas hospitalares combinando a precisão léxica do 
        **Okapi BM25** com a inteligência contextual de **Embeddings Semânticos** via **Reciprocal Rank Fusion (RRF)**.
        """
    )

    # Botões de consultas pré-configuradas para demonstração rápida
    st.markdown("**💡 Consultas Clínicas de Demonstração:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)

    preset_query = None
    if col_q1.button("🔤 Exame: CÓD-ECG-12D", use_container_width=True):
        preset_query = "CÓD-ECG-12D em arritmias"
    if col_q2.button("🧠 Sinônimo: infarto miocárdio", use_container_width=True):
        preset_query = "infarto agudo e dor precordial no peito"
    if col_q3.button("🧠 Sinônimo: AVC / Derrame", use_container_width=True):
        preset_query = "derrame cerebral e uso de trombolíticos"
    if col_q4.button("⚡ Híbrido: PCR Código Azul", use_container_width=True):
        preset_query = "parada cardiorrespiratória e choque no código azul"

    default_text = (
        preset_query
        if preset_query
        else "infarto agudo e eletrocardiograma CÓD-ECG-12D"
    )

    query_input = st.text_input(
        "🔍 Digite o termo médico, sintoma ou código clínico:",
        value=default_text,
        placeholder="Ex: CÓD-ECG-12D, infarto, crise hipertensiva, AVC trombolítico...",
    )

    if not query_input.strip():
        st.info(
            "Digite uma consulta acima para realizar a busca nos protocolos médicos."
        )
        return

    # Executa o Pipeline Unificado
    data = run_search_pipeline(
        query=query_input,
        k1=k1,
        b=b,
        alpha=alpha,
        enable_cross_encoder=enable_cross_encoder,
        corpus=st.session_state.corpus,
        preprocessor=st.session_state.preprocessor,
        bm25_engine=st.session_state.bm25_engine,
        semantic_engine=st.session_state.semantic_engine,
        rrf_engine=st.session_state.rrf_engine,
        reranker=st.session_state.reranker,
    )

    # Banner de Métricas Rápidas
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("⏱️ Latência", f"{data['latency_ms']:.1f} ms")
    col_m2.metric("⚖️ Peso BM25 (α)", f"{alpha:.2f}")
    col_m3.metric("🧠 Peso Semântico (1-α)", f"{1.0 - alpha:.2f}")
    col_m4.metric("✨ Cross-Encoder", "Ativo" if enable_cross_encoder else "Desativado")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔤 1. Busca Léxica (Okapi BM25)",
            "🧠 2. Busca Semântica (Embeddings)",
            "⚡ 3. Busca Híbrida (RRF)",
            "📊 4. Matriz Comparativa & Diagnóstico",
        ]
    )

    with tab1:
        st.subheader("🔤 Resultados do Motor Léxico Okapi BM25")
        st.caption(
            f"Calibração atual: k₁ = {k1}, b = {b}. Prioriza termos exatos e códigos literais."
        )
        for r in data["bm25_results"]:
            _render_document_card(r, mode="bm25", query=query_input)

    with tab2:
        st.subheader("🧠 Resultados do Motor Semântico Vetorial")
        st.caption(
            "Embeddings Densos com Similaridade de Cosseno. Captura sinônimos e intenção clínica."
        )
        for r in data["semantic_results"]:
            _render_document_card(r, mode="semantic", query=query_input)

    with tab3:
        st.subheader("⚡ Resultados da Fusão Híbrida (Reciprocal Rank Fusion)")
        st.caption(
            f"Fórmula: Score_RRF = {alpha:.2f} × [1 / (60 + Rank_BM25)] + {1.0 - alpha:.2f} × [1 / (60 + Rank_Semântico)]"
        )
        for r in data["hybrid_results"]:
            _render_document_card(r, mode="hybrid", query=query_input)

    with tab4:
        st.subheader("📊 Diagnóstico Comparativo & Análise de Desempenho")
        st.markdown(
            "Comparação lado a lado das posições e notas atribuídas por cada abordagem."
        )

        # Tabela Comparativa Formatada
        df = data["comparison_df"]
        st.dataframe(
            df[
                [
                    "ID",
                    "Título",
                    "Categoria",
                    "Rank BM25",
                    "Score BM25",
                    "Rank Semântico",
                    "Score Semântico",
                    "Rank Híbrido RRF",
                    "Score RRF",
                    "Score Cross-Encoder",
                    "Rank Shift (|BM25 - Semântico|)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Gráficos Visuais com Plotly
        if px is not None and go is not None:
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("##### 📈 Comparativo de Posições (Rank Shift)")
                # Gráfico de linhas paralelas de ranking
                fig_rank = go.Figure()
                for idx, row in df.iterrows():
                    fig_rank.add_trace(
                        go.Scatter(
                            x=[
                                "BM25 (Léxico)",
                                "Semântico (Vetorial)",
                                "Híbrido (RRF)",
                            ],
                            y=[
                                row["Rank BM25"],
                                row["Rank Semântico"],
                                row["Rank Híbrido RRF"],
                            ],
                            mode="lines+markers",
                            name=f"{row['ID']}: {row['Título'][:20]}...",
                            hovertemplate="<b>%{x}</b><br>Rank: #%{y}<extra></extra>",
                        )
                    )
                fig_rank.update_layout(
                    yaxis=dict(
                        autorange="reversed", title="Posição no Ranking (1º é melhor)"
                    ),
                    xaxis=dict(title="Motor de Recuperação"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=360,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.4,
                        xanchor="center",
                        x=0.5,
                    ),
                )
                st.plotly_chart(fig_rank, use_container_width=True)

            with col_g2:
                st.markdown("##### 🏆 Pontuação de Relevância RRF por Documento")
                fig_bar = px.bar(
                    df,
                    x="ID",
                    y="Score RRF",
                    color="Score RRF",
                    color_continuous_scale="Viridis",
                    hover_data=["Título", "Rank BM25", "Rank Semântico"],
                    labels={"Score RRF": "Pontuação RRF", "ID": "Documento"},
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360)
                st.plotly_chart(fig_bar, use_container_width=True)

        # Diagnóstico Inteligente Explicativo
        st.markdown("##### 🩺 Diagnóstico do Caso de Teste")
        top_hybrid = data["hybrid_results"][0]
        top_bm25 = data["bm25_results"][0]
        top_sem = data["semantic_results"][0]

        if top_hybrid.doc_id == top_bm25.doc_id == top_sem.doc_id:
            st.success(
                f"**Consenso Total**: Tanto o motor léxico quanto o semântico concordaram que o **{top_hybrid.doc_id} ({top_hybrid.title})** é a diretriz mais relevante."
            )
        elif top_bm25.doc_id != top_sem.doc_id:
            st.warning(
                f"**Divergência Resolvida pela Busca Híbrida**:\n"
                f"- O BM25 priorizou o **{top_bm25.doc_id}** (Rank Léxico #1).\n"
                f"- O Semântico priorizou o **{top_sem.doc_id}** (Rank Semântico #1).\n"
                f"- A Fusão RRF unificou os sinais posicionando o **{top_hybrid.doc_id} ({top_hybrid.title})** como melhor resposta balanceada."
            )


if __name__ == "__main__":
    main()
