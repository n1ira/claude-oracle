# Claude Oracle 개선 계획서
## 로컬 RAG 기반 프로젝트 인텔리전스 시스템

> **문서 버전**: 1.0
> **작성일**: 2024-12-05
> **상태**: 계획 단계

---

## 목차

1. [개요](#1-개요)
2. [아키텍처 설계](#2-아키텍처-설계)
3. [데이터 모델](#3-데이터-모델)
4. [신규 모듈 설계](#4-신규-모듈-설계)
5. [신규 CLI 명령어](#5-신규-cli-명령어)
6. [워크플로우 상세](#6-워크플로우-상세)
7. [에스컬레이션 정책](#7-에스컬레이션-정책)
8. [데이터 저장소 상세](#8-데이터-저장소-상세)
9. [비용 추정](#9-비용-추정)
10. [구현 로드맵](#10-구현-로드맵)
11. [성공 기준](#11-성공-기준)
12. [위험 및 완화](#12-위험-및-완화)

---

## 1. 개요

### 1.1 현재 상태

```
Claude Oracle v1.0
├── Gemini 3 Pro를 Oracle로 활용
├── 5개 교환 히스토리 (텍스트만)
├── FULLAUTO_CONTEXT.md로 수동 컨텍스트 관리
└── 임베딩/RAG 없음
```

### 1.2 목표 상태

```
Claude Oracle v2.0
├── 로컬 RAG 시스템 내장
├── 프로젝트 인텔리전스 자동 생성/유지
├── 임베딩 기반 의미 검색
├── 멀티-에이전트 감독 체계
└── 변경 영향 분석 자동화
```

### 1.3 핵심 가치

| 가치 | 설명 |
|------|------|
| **컨텍스트 지속성** | 세션/압축 후에도 프로젝트 이해 유지 |
| **영향 범위 파악** | 코드 변경 전 파급 효과 예측 |
| **비용 최적화** | 초기=고품질(Opus), 유지=저비용(Haiku) |
| **자동 동기화** | 코드 변경 시 RAG 자동 업데이트 |

### 1.4 기술 스택

| 구성요소 | 기술 |
|----------|------|
| 임베딩 | Google text-embedding-004 |
| 벡터 DB | ChromaDB |
| 메타데이터 | SQLite |
| 초기 분석 | Claude Opus + Gemini 3 Pro |
| 유지보수 | Claude Haiku (병렬 세션) |

---

## 2. 아키텍처 설계

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Claude Oracle v2.0                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Layer 1: CLI Interface                        │    │
│  │  oracle ask | oracle imagine | oracle rag | oracle analyze       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────▼───────────────────────────────┐    │
│  │                    Layer 2: Core Services                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ Query Engine │  │ RAG Engine   │  │ Sync Engine  │           │    │
│  │  │ (기존)       │  │ (신규)       │  │ (신규)       │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────▼───────────────────────────────┐    │
│  │                    Layer 3: Data Stores                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ History      │  │ Vector DB    │  │ Metadata     │           │    │
│  │  │ (기존)       │  │ (신규)       │  │ (신규)       │           │    │
│  │  │ JSON files   │  │ ChromaDB     │  │ SQLite       │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────▼───────────────────────────────┐    │
│  │                    Layer 4: External APIs                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ Gemini       │  │ Gemini       │  │ Claude       │           │    │
│  │  │ Generate     │  │ Embedding    │  │ (Host)       │           │    │
│  │  │ (기존)       │  │ (신규)       │  │              │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 에이전트 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        멀티-에이전트 체계                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Phase 1: 초기 분석] ─────────────────────────────────────────────────  │
│                                                                          │
│      ┌─────────────────┐         ┌─────────────────┐                    │
│      │  Gemini 3 Pro   │◄───────►│  Claude Opus    │                    │
│      │  (전략/설계)     │         │  (분석/구현)     │                    │
│      └────────┬────────┘         └────────┬────────┘                    │
│               │                           │                              │
│               └───────────┬───────────────┘                              │
│                           ▼                                              │
│               ┌─────────────────────┐                                   │
│               │  RAG 초기 구축       │                                   │
│               │  - 전체 코드 분석    │                                   │
│               │  - 청크 생성        │                                   │
│               │  - 임베딩 생성      │                                   │
│               │  - 메타데이터 구축   │                                   │
│               └─────────────────────┘                                   │
│                                                                          │
│  [Phase 2: 지속 유지] ─────────────────────────────────────────────────  │
│                                                                          │
│      ┌─────────────────┐         ┌─────────────────┐                    │
│      │  Haiku Watcher  │         │  Haiku Validator│                    │
│      │  (세션 A)       │         │  (세션 B)       │                    │
│      │                 │         │                 │                    │
│      │  • 파일 변경 감지│         │  • 품질 검증    │                    │
│      │  • 청크 업데이트 │         │  • 일관성 검사  │                    │
│      │  • 임베딩 갱신  │         │  • 이상 탐지    │                    │
│      └────────┬────────┘         └────────┬────────┘                    │
│               │                           │                              │
│               └───────────┬───────────────┘                              │
│                           ▼                                              │
│               ┌─────────────────────┐                                   │
│               │  에스컬레이션 조건   │                                   │
│               │  충족 시 Opus 재투입 │                                   │
│               └─────────────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 RAG 작동 원리

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG 검색 흐름                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [인덱싱 시점]                                                           │
│                                                                          │
│  코드/문서 ──► 청킹 ──► 임베딩 API ──► 벡터 ──► ChromaDB 저장           │
│                         (text-embedding-004)                            │
│                                                                          │
│  ※ 원본 텍스트도 함께 저장                                              │
│                                                                          │
│  [검색 시점]                                                             │
│                                                                          │
│  질문 ──► 임베딩 API ──► 질문 벡터 ──► ChromaDB 유사도 검색             │
│                                              │                          │
│                                              ▼                          │
│                                     유사한 청크의 **원본 텍스트** 반환   │
│                                              │                          │
│                                              ▼                          │
│                                     LLM 프롬프트에 컨텍스트로 추가       │
│                                              │                          │
│                                              ▼                          │
│                                     Claude/Gemini가 텍스트 읽고 답변    │
│                                                                          │
│  ※ LLM은 벡터를 받지 않음 - 검색된 텍스트만 받음                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 데이터 모델

### 3.1 청크 스키마

```python
# rag/schemas.py

from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class ChunkType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    CONCEPT = "concept"      # 설계 철학, 규칙 등
    DEPENDENCY = "dependency"
    TODO = "todo"

class Chunk(BaseModel):
    # 식별
    chunk_id: str                    # UUID
    type: ChunkType
    name: str                        # 함수명, 클래스명 등

    # 위치
    file_path: str
    line_start: int
    line_end: int

    # 내용
    content: str                     # 원본 코드/텍스트
    summary: str                     # LLM 생성 요약
    embedding: List[float]           # 768차원 벡터

    # 관계
    depends_on: List[str]            # chunk_id 목록
    depended_by: List[str]
    related_concepts: List[str]

    # 메타
    keywords: List[str]
    complexity: Literal["low", "medium", "high"]
    status: Literal["stable", "experimental", "deprecated"]

    # 버전
    code_hash: str                   # 코드 내용의 해시
    commit_hash: str
    created_by: Literal["opus", "haiku"]
    created_at: datetime
    updated_at: datetime
```

### 3.2 프로젝트 메타데이터

```python
# rag/schemas.py

class ProjectMeta(BaseModel):
    # 기본 정보
    project_id: str
    project_name: str
    project_path: str

    # 분석 상태
    last_full_analysis: datetime
    last_incremental_update: datetime
    analysis_version: str            # "2.0.0"

    # 통계
    total_chunks: int
    total_files: int
    total_functions: int
    total_classes: int

    # 아키텍처 요약 (Opus가 생성)
    architecture_summary: str
    design_philosophy: str
    key_constraints: List[str]
    known_limitations: List[str]

    # 품질
    rag_quality_score: float         # 0-100
    last_validation: datetime

    # 로드맵
    completion_status: Dict[str, bool]
    tech_debt: List[str]
    roadmap: List[str]
```

### 3.3 변경 로그

```python
# rag/schemas.py

class ChangeLog(BaseModel):
    log_id: str
    timestamp: datetime
    trigger: Literal["git_commit", "file_watch", "manual"]

    # 변경 내용
    commit_hash: Optional[str]
    files_changed: List[str]
    chunks_created: List[str]
    chunks_updated: List[str]
    chunks_deleted: List[str]

    # 처리 정보
    processed_by: Literal["haiku_watcher", "haiku_validator", "opus"]
    processing_time_ms: int
    tokens_used: int

    # 상태
    status: Literal["success", "partial", "failed", "escalated"]
    error_message: Optional[str]
    escalation_reason: Optional[str]
```

---

## 4. 신규 모듈 설계

### 4.1 파일 구조

```
oracle.py (기존, 수정)
schemas.py (기존, 확장)

rag/
├── __init__.py
├── engine.py              # RAG 핵심 엔진
├── embedder.py            # 임베딩 처리
├── chunker.py             # 코드 청킹
├── indexer.py             # 인덱싱/검색
├── analyzer.py            # 코드 분석 (AST)
├── schemas.py             # RAG 전용 스키마
└── db/
    ├── __init__.py
    ├── vector_store.py    # ChromaDB 래퍼
    └── meta_store.py      # SQLite 래퍼

agents/
├── __init__.py
├── watcher.py             # Haiku Watcher 에이전트
├── validator.py           # Haiku Validator 에이전트
└── orchestrator.py        # 에이전트 조율

commands/
├── fullauto.md (기존)
├── rag-init.md            # 초기 RAG 구축 명령
├── rag-query.md           # RAG 쿼리 명령
└── rag-sync.md            # 수동 동기화 명령
```

### 4.2 embedder.py

```python
"""임베딩 생성 모듈"""

from google import genai
from typing import List
import hashlib

class GeminiEmbedder:
    """Google text-embedding-004를 사용한 임베딩 생성"""

    MODEL = "text-embedding-004"
    DIMENSION = 768
    BATCH_SIZE = 100

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._cache = {}  # 중복 임베딩 방지

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._cache:
            return self._cache[cache_key]

        response = self.client.models.embed_content(
            model=self.MODEL,
            content=text
        )

        embedding = response.embedding
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 (비용 최적화)"""
        results = []
        uncached = []
        uncached_indices = []

        # 캐시 확인
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached.append(text)
                uncached_indices.append(i)

        # 미싱만 API 호출
        if uncached:
            # 배치 처리
            for batch_start in range(0, len(uncached), self.BATCH_SIZE):
                batch = uncached[batch_start:batch_start + self.BATCH_SIZE]
                # API 호출 로직
                ...

        # 결과 정렬 및 반환
        ...

    def embed_chunk(self, chunk: 'Chunk') -> List[float]:
        """청크용 임베딩 (요약 + 키워드 결합)"""
        combined = f"{chunk.summary}\n\nKeywords: {', '.join(chunk.keywords)}"
        return self.embed_text(combined)
```

### 4.3 chunker.py

```python
"""코드 청킹 모듈"""

import ast
from pathlib import Path
from typing import List, Generator

class CodeChunker:
    """AST 기반 의미 단위 코드 분할"""

    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.go', '.rs'}

    def __init__(self, llm_client):
        self.llm = llm_client  # 요약 생성용

    def chunk_file(self, file_path: Path) -> Generator['Chunk', None, None]:
        """파일을 의미 단위로 분할"""

        ext = file_path.suffix
        content = file_path.read_text()

        if ext == '.py':
            yield from self._chunk_python(file_path, content)
        elif ext in {'.js', '.ts'}:
            yield from self._chunk_javascript(file_path, content)
        else:
            yield from self._chunk_generic(file_path, content)

    def _chunk_python(self, path: Path, content: str) -> Generator:
        """Python AST 기반 청킹"""
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                yield self._create_function_chunk(path, node, content)
            elif isinstance(node, ast.ClassDef):
                yield self._create_class_chunk(path, node, content)

    def _create_function_chunk(self, path, node, content) -> 'Chunk':
        """함수 청크 생성"""
        source = ast.get_source_segment(content, node)

        # LLM으로 요약 생성 (Haiku 사용)
        summary = self.llm.summarize_code(source)

        # 키워드 추출
        keywords = self._extract_keywords(node, source)

        return Chunk(
            type=ChunkType.FUNCTION,
            name=node.name,
            file_path=str(path),
            line_start=node.lineno,
            line_end=node.end_lineno,
            content=source,
            summary=summary,
            keywords=keywords,
            # ... 기타 필드
        )

    def _extract_keywords(self, node, source: str) -> List[str]:
        """함수에서 키워드 추출"""
        keywords = [node.name]

        # 파라미터명
        for arg in node.args.args:
            keywords.append(arg.arg)

        # 호출하는 함수명
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    keywords.append(child.func.id)

        return list(set(keywords))
```

### 4.4 engine.py

```python
"""RAG 핵심 엔진"""

from pathlib import Path
from typing import List, Optional, Tuple
from .embedder import GeminiEmbedder
from .chunker import CodeChunker
from .indexer import VectorIndexer
from .db.vector_store import ChromaStore
from .db.meta_store import MetaStore

class RAGEngine:
    """RAG 시스템 통합 엔진"""

    def __init__(self, project_path: str, config: dict):
        self.project_path = Path(project_path)
        self.config = config

        # 컴포넌트 초기화
        self.embedder = GeminiEmbedder(config['gemini_api_key'])
        self.chunker = CodeChunker(config['llm_client'])
        self.vector_store = ChromaStore(self.project_path / '.rag' / 'vectors')
        self.meta_store = MetaStore(self.project_path / '.rag' / 'meta.db')
        self.indexer = VectorIndexer(self.vector_store, self.embedder)

    # ─────────────────────────────────────────────────────────
    # 초기 분석 (Opus + Gemini)
    # ─────────────────────────────────────────────────────────

    def initialize(self, analyzer_model: str = "opus") -> 'AnalysisReport':
        """프로젝트 초기 분석 및 RAG 구축"""

        report = AnalysisReport()

        # 1. 파일 스캔
        files = self._scan_project_files()
        report.files_found = len(files)

        # 2. 청킹
        all_chunks = []
        for file in files:
            chunks = list(self.chunker.chunk_file(file))
            all_chunks.extend(chunks)
        report.chunks_created = len(all_chunks)

        # 3. 의존성 분석
        self._analyze_dependencies(all_chunks)

        # 4. 임베딩 생성
        for chunk in all_chunks:
            chunk.embedding = self.embedder.embed_chunk(chunk)

        # 5. 저장
        self.vector_store.add_chunks(all_chunks)
        self.meta_store.save_chunks_meta(all_chunks)

        # 6. 프로젝트 메타 생성 (Opus)
        project_meta = self._generate_project_meta(all_chunks, analyzer_model)
        self.meta_store.save_project_meta(project_meta)

        report.success = True
        return report

    # ─────────────────────────────────────────────────────────
    # 검색 (Query Time)
    # ─────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> 'QueryResult':
        """자연어 질문으로 관련 청크 검색"""

        # 1. 질문 임베딩
        query_embedding = self.embedder.embed_text(question)

        # 2. 벡터 검색
        similar_chunks = self.vector_store.search(
            embedding=query_embedding,
            top_k=top_k
        )

        # 3. 메타데이터 보강
        enriched = self._enrich_with_metadata(similar_chunks)

        # 4. 관련 청크 확장 (의존성 따라)
        expanded = self._expand_related(enriched)

        return QueryResult(
            chunks=expanded,
            query=question,
            total_tokens=self._estimate_tokens(expanded)
        )

    def get_impact_analysis(self, file_path: str) -> 'ImpactReport':
        """파일 변경 시 영향 범위 분석"""

        # 해당 파일의 청크들
        file_chunks = self.meta_store.get_chunks_by_file(file_path)

        # 이 청크들에 의존하는 다른 청크들
        impacted = set()
        for chunk in file_chunks:
            dependents = self.meta_store.get_dependents(chunk.chunk_id)
            impacted.update(dependents)

        return ImpactReport(
            source_file=file_path,
            source_chunks=file_chunks,
            impacted_chunks=list(impacted),
            impacted_files=list(set(c.file_path for c in impacted))
        )

    # ─────────────────────────────────────────────────────────
    # 증분 업데이트 (Haiku)
    # ─────────────────────────────────────────────────────────

    def update_file(self, file_path: str) -> 'UpdateReport':
        """단일 파일 변경 처리"""

        # 1. 기존 청크 조회
        old_chunks = self.meta_store.get_chunks_by_file(file_path)

        # 2. 새 청크 생성
        new_chunks = list(self.chunker.chunk_file(Path(file_path)))

        # 3. 차이 분석
        to_delete, to_update, to_create = self._diff_chunks(old_chunks, new_chunks)

        # 4. 적용
        self.vector_store.delete_chunks(to_delete)
        self.vector_store.update_chunks(to_update)
        self.vector_store.add_chunks(to_create)

        # 5. 메타 업데이트
        self.meta_store.update_chunks_meta(new_chunks)

        return UpdateReport(
            deleted=len(to_delete),
            updated=len(to_update),
            created=len(to_create)
        )

    def _scan_project_files(self) -> List[Path]:
        """프로젝트 파일 스캔"""
        files = []
        ignore_patterns = self.config.get('ignore_patterns', [])

        for ext in self.chunker.SUPPORTED_EXTENSIONS:
            for file in self.project_path.rglob(f"*{ext}"):
                # .gitignore 및 설정된 패턴 적용
                if not self._should_ignore(file, ignore_patterns):
                    files.append(file)

        return files

    def _analyze_dependencies(self, chunks: List['Chunk']):
        """청크 간 의존성 분석"""
        # 함수/클래스명 → chunk_id 매핑
        name_to_chunk = {c.name: c.chunk_id for c in chunks}

        for chunk in chunks:
            # AST에서 호출/참조 추출
            calls = self._extract_calls(chunk.content)

            for call in calls:
                if call in name_to_chunk:
                    target_id = name_to_chunk[call]
                    chunk.depends_on.append(target_id)
                    # 역방향도 설정
                    target_chunk = next(c for c in chunks if c.chunk_id == target_id)
                    target_chunk.depended_by.append(chunk.chunk_id)
```

---

## 5. 신규 CLI 명령어

### 5.1 명령어 목록

```bash
# 초기 분석 (1회, Opus + Gemini)
oracle rag init [--full] [--model opus]

# RAG 쿼리
oracle rag query "에러 처리 로직은 어디?"
oracle rag query --impact src/oracle.py  # 영향 분석

# 동기화
oracle rag sync                          # 전체 동기화
oracle rag sync --file src/oracle.py     # 단일 파일

# 상태 확인
oracle rag status
oracle rag stats

# Watcher 실행 (백그라운드)
oracle rag watch --daemon

# 검증
oracle rag validate

# 에스컬레이션 처리
oracle rag escalate
```

### 5.2 oracle.py 수정 사항

```python
# oracle.py에 추가될 서브파서

# rag 명령 그룹
rag_parser = subparsers.add_parser("rag", help="RAG system management")
rag_subparsers = rag_parser.add_subparsers(dest="rag_command")

# init
rag_init = rag_subparsers.add_parser("init", help="Initialize RAG")
rag_init.add_argument("--full", action="store_true", help="Full analysis")
rag_init.add_argument("--model", default="opus", choices=["opus", "haiku"])

# query
rag_query = rag_subparsers.add_parser("query", help="Query RAG")
rag_query.add_argument("question", nargs="?", help="Natural language question")
rag_query.add_argument("--impact", help="Analyze impact of file changes")
rag_query.add_argument("--top-k", type=int, default=5)

# sync
rag_sync = rag_subparsers.add_parser("sync", help="Sync RAG with code")
rag_sync.add_argument("--file", help="Sync single file only")

# watch
rag_watch = rag_subparsers.add_parser("watch", help="Watch for changes")
rag_watch.add_argument("--daemon", action="store_true")

# status
rag_status = rag_subparsers.add_parser("status", help="Show RAG status")

# validate
rag_validate = rag_subparsers.add_parser("validate", help="Validate RAG integrity")

# escalate
rag_escalate = rag_subparsers.add_parser("escalate", help="Process pending escalations")
```

---

## 6. 워크플로우 상세

### 6.1 초기 분석 워크플로우

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    oracle rag init --full                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Step 1: 프로젝트 스캔                                                   │
│  ────────────────────                                                   │
│  • 지원 파일 확장자 필터링 (.py, .js, .ts, .go, .rs)                    │
│  • .gitignore 적용                                                      │
│  • 파일 목록 생성                                                        │
│                                                                          │
│  Step 2: 코드 파싱 (AST)                                                │
│  ────────────────────                                                   │
│  • 함수, 클래스, 모듈 추출                                               │
│  • 의존성 그래프 구축                                                    │
│  • 호출 관계 분석                                                        │
│                                                                          │
│  Step 3: 청크 생성                                                       │
│  ────────────────────                                                   │
│  • 코드 단위별 청크 분할                                                 │
│  • Opus로 각 청크 요약 생성                                              │
│  • 키워드 추출                                                           │
│                                                                          │
│  Step 4: 임베딩                                                          │
│  ────────────────────                                                   │
│  • text-embedding-004로 벡터화                                          │
│  • 배치 처리로 비용 최적화                                               │
│                                                                          │
│  Step 5: 프로젝트 메타 생성 (Gemini + Opus)                             │
│  ────────────────────                                                   │
│  • 아키텍처 요약                                                         │
│  • 설계 철학 추론                                                        │
│  • 제약 사항 정리                                                        │
│  • 완성도 평가                                                           │
│  • 기술 부채 식별                                                        │
│                                                                          │
│  Step 6: 저장                                                            │
│  ────────────────────                                                   │
│  • ChromaDB에 벡터 저장                                                  │
│  • SQLite에 메타데이터 저장                                              │
│  • 분석 리포트 출력                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 쿼리 워크플로우

```
┌─────────────────────────────────────────────────────────────────────────┐
│               oracle rag query "에러 처리 로직은 어디?"                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 질문 입력                                                            │
│         │                                                                │
│         ▼                                                                │
│  2. text-embedding-004로 질문 벡터화                                    │
│         │                                                                │
│         ▼                                                                │
│  3. ChromaDB에서 코사인 유사도 검색 (top-k)                             │
│         │                                                                │
│         ▼                                                                │
│  4. 검색 결과:                                                           │
│     • log_error() - 유사도 0.92                                         │
│     • ask_oracle() try-except - 유사도 0.87                             │
│     • imagine() except block - 유사도 0.84                              │
│         │                                                                │
│         ▼                                                                │
│  5. 관계 확장 (의존성 그래프 탐색)                                       │
│         │                                                                │
│         ▼                                                                │
│  6. 결과 출력:                                                           │
│     ## 관련 코드                                                         │
│     ### 1. log_error (oracle.py:134-138) [유사도: 92%]                  │
│     에러 메시지를 파일과 콘솔에 로깅                                      │
│                                                                          │
│     ## 영향 범위                                                         │
│     - log_error는 12개 함수에서 호출됨                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Watcher 워크플로우 (Haiku)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    oracle rag watch --daemon                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  트리거: Git Hook (post-commit) 또는 File System Watcher                │
│                             │                                            │
│                             ▼                                            │
│  변경 감지:                                                              │
│  • src/oracle.py 수정됨                                                 │
│  • src/new_module.py 추가됨                                             │
│                             │                                            │
│                             ▼                                            │
│  복잡도 판단 (Haiku):                                                   │
│                                                                          │
│  ┌─────────────────────────┬─────────────────────────┐                  │
│  │ 단순 변경               │ 복잡한 변경              │                  │
│  │ • 기존 함수 내용 수정   │ • 새 모듈 추가          │                  │
│  │ • 변수명 변경           │ • 아키텍처 패턴 변경    │                  │
│  │ • 주석 추가             │ • 10개+ 함수 동시 변경  │                  │
│  │         │               │         │               │                  │
│  │         ▼               │         ▼               │                  │
│  │   Haiku 처리            │   Opus 에스컬레이션     │                  │
│  └─────────────────────────┴─────────────────────────┘                  │
│                                                                          │
│  Haiku 처리:                                                            │
│  1. 영향 청크 식별                                                       │
│  2. 청크 재생성                                                          │
│  3. 임베딩 갱신                                                          │
│  4. 로그 기록                                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 에스컬레이션 정책

### 7.1 자동 에스컬레이션 조건

```python
# agents/orchestrator.py

class EscalationPolicy:
    """Haiku → Opus 에스컬레이션 조건"""

    THRESHOLDS = {
        # 파일 수 기준
        'max_files_changed': 10,

        # 복잡도 기준
        'new_module_added': True,        # 새 모듈 = 즉시 에스컬레이션
        'new_class_added': True,         # 새 클래스 = 즉시 에스컬레이션
        'architecture_pattern_change': True,

        # 품질 기준
        'min_quality_score': 80,         # 80% 미만이면 에스컬레이션
        'max_validation_failures': 3,    # 3회 연속 실패

        # 의존성 기준
        'dependency_depth_change': 2,    # 의존성 깊이 2단계 이상 변경
    }

    def should_escalate(self, change_report) -> Tuple[bool, str]:
        """에스컬레이션 필요 여부 판단"""

        if change_report.new_modules:
            return (True, "새 모듈 추가됨")

        if change_report.files_changed > self.THRESHOLDS['max_files_changed']:
            return (True, f"{change_report.files_changed}개 파일 동시 변경")

        if change_report.quality_score < self.THRESHOLDS['min_quality_score']:
            return (True, f"품질 점수 {change_report.quality_score}% < 80%")

        return (False, None)
```

### 7.2 에스컬레이션 처리

```
에스컬레이션 발생 시:

1. 에스컬레이션 플래그 저장
   .rag/escalation/pending.json
   {
     "reason": "새 모듈 추가됨",
     "files": ["src/new_module.py"],
     "timestamp": "2024-12-05T10:30:00",
     "haiku_analysis": "팩토리 패턴으로 보이나 확신 없음"
   }

2. 사용자에게 알림 (다음 oracle 명령 시)
   "⚠️ RAG 에스컬레이션 대기 중: 새 모듈 추가됨"
   "oracle rag escalate 실행하여 Opus 분석 시작"

3. oracle rag escalate 실행 시
   • Opus가 해당 변경 분석
   • 청크 재생성
   • 프로젝트 메타 업데이트
   • 품질 검증
```

---

## 8. 데이터 저장소 상세

### 8.1 디렉토리 구조

```
project/
└── .rag/
    ├── config.yaml                # RAG 설정
    │
    ├── vectors/                   # ChromaDB
    │   ├── chroma.sqlite3
    │   └── index/
    │
    ├── meta/                      # SQLite 메타데이터
    │   └── meta.db
    │
    ├── cache/                     # 임베딩 캐시
    │   └── embeddings.cache
    │
    ├── logs/                      # 변경 로그
    │   ├── changes.jsonl
    │   └── validations.jsonl
    │
    ├── escalation/                # 에스컬레이션 대기
    │   └── pending.json
    │
    └── exports/                   # 내보내기
        └── intelligence.md        # 사람이 읽을 수 있는 요약
```

### 8.2 config.yaml

```yaml
# .rag/config.yaml

version: "2.0"

# 임베딩 설정
embedding:
  model: "text-embedding-004"
  batch_size: 100
  cache_enabled: true

# 청킹 설정
chunking:
  supported_extensions:
    - .py
    - .js
    - .ts
    - .go
    - .rs
    - .java
  ignore_patterns:
    - "**/test_*.py"
    - "**/__pycache__/**"
    - "**/node_modules/**"
    - "**/.git/**"
  max_chunk_lines: 100
  min_chunk_lines: 5

# 검색 설정
search:
  default_top_k: 5
  expand_related: true
  max_expansion_depth: 2

# 에이전트 설정
agents:
  watcher:
    enabled: true
    trigger: "git_hook"  # git_hook | file_watch | manual
    model: "haiku"
  validator:
    enabled: true
    interval_hours: 1
    model: "haiku"
  escalation:
    auto_run: false      # true면 자동 Opus 실행
    model: "opus"

# 에스컬레이션 임계값
escalation:
  max_files_changed: 10
  min_quality_score: 80
  new_module_triggers: true
```

---

## 9. 비용 추정

### 9.1 초기 분석 (1회)

| 항목 | 모델 | 토큰 (추정) | 비용 |
|------|------|-------------|------|
| 코드 요약 생성 | Opus | ~200K input, ~50K output | ~$6 |
| 프로젝트 메타 | Opus + Gemini | ~100K | ~$3 |
| 임베딩 | text-embedding-004 | ~500K chars | ~$0.05 |
| **초기 합계** | | | **~$9-10** |

### 9.2 월간 유지 (Haiku)

| 항목 | 빈도 | 토큰/회 | 월 비용 |
|------|------|---------|---------|
| Watcher 업데이트 | ~100회/일 | ~2K | ~$6 |
| Validator 검증 | ~24회/일 | ~5K | ~$3 |
| 임베딩 갱신 | ~100회/일 | ~10K chars | ~$0.01 |
| **월간 합계** | | | **~$9-10/월** |

### 9.3 에스컬레이션 (필요시)

| 항목 | 빈도 (추정) | 비용/회 | 월 비용 |
|------|-------------|---------|---------|
| Opus 재분석 | ~2회/월 | ~$5 | ~$10 |

### 9.4 총 비용 요약

```
초기 구축:    ~$10 (1회성)
월간 유지:    ~$10-20/월
─────────────────────────
연간 총:      ~$130-250/년
```

---

## 10. 구현 로드맵

### Phase 1: 핵심 인프라 (1주)

- [ ] `rag/` 디렉토리 구조 생성
- [ ] `embedder.py` - Gemini 임베딩 래퍼
- [ ] `db/vector_store.py` - ChromaDB 통합
- [ ] `db/meta_store.py` - SQLite 메타데이터
- [ ] 기본 `config.yaml` 스키마

### Phase 2: 청킹 & 분석 (1주)

- [ ] `chunker.py` - AST 기반 코드 분할
- [ ] `analyzer.py` - 의존성 분석
- [ ] `rag/schemas.py` - Chunk, ProjectMeta 등
- [ ] `engine.py` 기본 구조

### Phase 3: 초기 분석 기능 (1주)

- [ ] `engine.py` `initialize()` 구현
- [ ] `oracle rag init` 명령어
- [ ] Opus 연동 (요약, 메타 생성)
- [ ] 분석 리포트 출력

### Phase 4: 쿼리 기능 (1주)

- [ ] `indexer.py` - 벡터 검색
- [ ] `engine.py` `query()` 구현
- [ ] `engine.py` `get_impact_analysis()` 구현
- [ ] `oracle rag query` 명령어

### Phase 5: 자동 동기화 (1주)

- [ ] `agents/watcher.py`
- [ ] Git hook 연동
- [ ] `engine.py` `update_file()` 구현
- [ ] `oracle rag sync` 명령어

### Phase 6: 검증 & 에스컬레이션 (1주)

- [ ] `agents/validator.py`
- [ ] `agents/orchestrator.py`
- [ ] 에스컬레이션 정책 구현
- [ ] `oracle rag validate` 명령어

### Phase 7: 통합 & 테스트 (1주)

- [ ] 전체 워크플로우 테스트
- [ ] `fullauto.md` RAG 연동
- [ ] 문서화
- [ ] `install.sh` 업데이트

---

## 11. 성공 기준

| 기준 | 목표 |
|------|------|
| 초기 분석 시간 | < 5분 (중간 규모 프로젝트) |
| 쿼리 응답 시간 | < 3초 |
| RAG 품질 점수 | > 85% |
| 증분 업데이트 시간 | < 10초/파일 |
| 에스컬레이션 정확도 | > 90% |

---

## 12. 위험 및 완화

| 위험 | 영향도 | 완화 방안 |
|------|--------|-----------|
| 임베딩 API 비용 초과 | 중 | 캐싱, 배치 처리, 변경분만 갱신 |
| ChromaDB 성능 한계 | 중 | 샤딩, 또는 Qdrant 대체 |
| 청킹 품질 저하 | 고 | AST 파싱 + LLM 검증 병행 |
| 동기화 지연 | 저 | Git hook으로 즉시 트리거 |
| Opus 비용 폭증 | 중 | 에스컬레이션 임계값 조정 |
| 임베딩 모델 변경 | 중 | 추상화 레이어로 모델 교체 용이하게 |

---

## 부록 A: 의존성

### 신규 Python 패키지

```
chromadb>=0.4.0          # 벡터 데이터베이스
watchdog>=3.0.0          # 파일 시스템 감시
gitpython>=3.1.0         # Git 연동
```

### install.sh 수정 사항

```bash
# 기존 의존성에 추가
"$INSTALL_DIR/venv/bin/pip" install -q \
    chromadb \
    watchdog \
    gitpython
```

---

## 부록 B: 참고 자료

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Google AI Embedding Models](https://ai.google.dev/models/gemini)
- [AST Module - Python Docs](https://docs.python.org/3/library/ast.html)

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|-----------|
| 1.0 | 2024-12-05 | 최초 작성 |
