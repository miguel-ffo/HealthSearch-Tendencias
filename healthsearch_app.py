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
        "meus", "minha", "minhas", "na", "nas", "não", "nas", "no", "nos", "nossa",
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
            # Caso query vazia, retorna os documentos na ordem com score zero
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

        # Inicializa BM25Okapi com os parâmetros dinâmicos
        if BM25Okapi is not None:
            bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)
            raw_scores = bm25.get_scores(query_tokens)
        else:
            # Fallback manual de pontuação TF caso rank_bm25 não esteja carregado
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
