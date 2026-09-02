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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

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
            category="Emergência Cardiológica"
        ),
        Document(
            id="Doc 2",
            title="Guia de Farmacologia Cardíaca",
            content="O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio.",
            category="Farmacologia"
        ),
        Document(
            id="Doc 3",
            title="Diretriz de Hipertensão Arterial",
            content="A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI.",
            category="Terapia Intensiva"
        ),
        Document(
            id="Doc 4",
            title="Manual de AVC Isquêmico",
            content="O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas.",
            category="Neurologia de Emergência"
        ),
        Document(
            id="Doc 5",
            title="Protocolo de Reanimação RCR",
            content="Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul.",
            category="Suporte Avançado de Vida"
        ),
        Document(
            id="Doc 6",
            title="Procedimentos de UTI Geral",
            content="Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria.",
            category="Terapia Intensiva"
        )
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
        "a", "à", "ao", "aos", "as", "às", "com", "da", "das", "de", "do", "dos",
        "e", "é", "em", "entre", "era", "essa", "essas", "esse", "esses", "esta",
        "estas", "este", "estes", "eu", "foi", "foram", "fosse", "ha", "há",
        "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
        "meus", "minha", "minhas", "na", "nas", "não", "no", "nos", "nossa",
        "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela",
        "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem", "se",
        "seja", "sem", "ser", "será", "seu", "seus", "só", "sua", "suas", "também",
        "te", "tem", "têm", "tinha", "tu", "tua", "tuas", "um", "uma", "você", "vocês",
        "ate", "até", "deles", "delas", "dele", "dela"
    }

    # Regex para capturar palavras e códigos alfanuméricos com hífens (ex: CÓD-ECG-12D, 100mg)
    TOKEN_REGEX = re.compile(r'[a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]+(?:-[a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]+)*')

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
            
        return [t for t in raw_tokens if t not in self.PORTUGUESE_STOPWORDS and len(t) > 0]


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

    def __init__(self, corpus: MedicalCorpus, preprocessor: Optional[TextPreprocessor] = None):
        self.corpus = corpus
        self.preprocessor = preprocessor or TextPreprocessor()
        self.docs = self.corpus.get_documents()
        self.tokenized_corpus = [
            self.preprocessor.tokenize(f"{d.title} {d.content}")
            for d in self.docs
        ]

    def search(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[SearchResult]:
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
                    category=d.category
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

        # Cria lista de resultados ordenada por score decrescente
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
                    category=doc.category
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
            self._doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True)

    def search(self, query: str) -> List[SearchResult]:
        """Calcula a similaridade de cosseno entre a consulta e os documentos médicos."""
        if not query or not query.strip():
            return [
                SearchResult(
                    doc_id=d.id, score=0.0, rank=i + 1,
                    title=d.title, content=d.content, category=d.category
                )
                for i, d in enumerate(self.docs)
            ]

        if self.model is not None and self._doc_embeddings is not None and util is not None:
            query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            cos_scores = util.cos_sim(query_embedding, self._doc_embeddings)[0]
            scores_np = cos_scores.cpu().numpy()
        else:
            # Fallback heurístico resiliente
            scores_np = np.zeros(len(self.docs))
            q_words = set(query.lower().split())
            for i, d in enumerate(self.docs):
                d_words = set(f"{d.title} {d.content}".lower().split())
                overlap = len(q_words.intersection(d_words)) / max(len(q_words), 1)
                scores_np[i] = overlap

        # Normaliza scores no intervalo [0.0, 1.0]
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
                    category=doc.category
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
        k_rrf: int = 60
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
            doc_scores.append({
                "doc_id": doc_id,
                "score": rrf_score,
                "bm25_rank": r_bm25,
                "semantic_rank": r_sem,
                "title": ref_doc.title,
                "content": ref_doc.content,
                "category": ref_doc.category
            })

        # Ordenação com suporte aos limites de alpha (empate resolvido pelo motor prioritário)
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
                    semantic_rank=item["semantic_rank"]
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

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        """Aplica Cross-Encoder sobre os Top-K candidatos da busca híbrida."""
        if not candidates:
            return []

        top_candidates = [
            SearchResult(
                doc_id=c.doc_id, score=c.score, rank=c.rank,
                title=c.title, content=c.content, category=c.category,
                bm25_rank=c.bm25_rank, semantic_rank=c.semantic_rank,
                cross_score=c.cross_score
            )
            for c in candidates[:top_k]
        ]
        rest_candidates = [
            SearchResult(
                doc_id=c.doc_id, score=c.score, rank=c.rank,
                title=c.title, content=c.content, category=c.category,
                bm25_rank=c.bm25_rank, semantic_rank=c.semantic_rank,
                cross_score=None
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

        # Re-ordena os Top-K com base no Cross-Encoder score
        top_candidates.sort(key=lambda x: (x.cross_score if x.cross_score is not None else -1.0), reverse=True)

        full_results = []
        for rank_idx, item in enumerate(top_candidates + rest_candidates, start=1):
            item.rank = rank_idx
            full_results.append(item)

        return full_results


