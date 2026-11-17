"""
Exact + semantic result cache for the Text-to-SQL graph.

Lookup order:
  1. EXACT    - normalized request text (+ namespace/context) -> hit
  2. SEMANTIC - embedding similarity >= threshold, but ONLY if a "guard
                signature" matches (same numbers, same top/bottom/avg/sum/not
                intent, same entity/value tokens). This stops
                "top 3 categories" from matching "top 5 categories", or
                "orders in Delhi" from matching "orders in Mumbai".

Each entry stores the validated SQL and its result with two TTLs:
  - result_ttl (short): how long the stored RESULT is served as-is
  - sql_ttl   (long) : after result_ttl the SQL is re-executed on the DB
                       (no LLM call) to get fresh rows; after sql_ttl the
                       entry is dropped and the full pipeline runs again.

In-memory + thread-safe. Per-process only: on Cloud Run with several
instances each one has its own cache. Swap `_entries` for Redis if you need
a shared cache (the public methods are the only interface the graph uses).
"""
from __future__ import annotations

import re
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Text normalisation + guard signature
# --------------------------------------------------------------------------
_STOP = set(
    """a an the of in on for to by with from and is are was were be been being have has had
    do does did show me us give get find list display tell what which who whom how much
    please can could would you i we my our their its these those them it that this than then
    as at about into over under between sort sorted order ordered group grouped rank ranked
    compare comparison data information details record records row rows value values
    every each all any individual individually respective respectively corresponding
    separate separately also along together there they their some ones""".split()
)

_CANON: dict[str, str] = {}
for _canon, _words in {
    "HIGH": "top highest most best largest biggest greatest max maximum",
    "LOW": "bottom lowest least worst smallest fewest min minimum",
    "ASC": "asc ascending increasing",
    "DESC": "desc descending decreasing",
    "AVG": "avg average mean",
    "SUM": "sum total overall",
    "COUNT": "count number many",
    "NOT": "not no without except exclude excluding excluded non",
    "OR": "or either",
    "ONLY": "only just",
}.items():
    for _w in _words.split():
        _CANON[_w] = _canon

_NUMWORDS = {w: str(i) for i, w in enumerate("one two three four five six seven eight nine ten".split(), 1)}

_FILLER = re.compile(
    r"^(?:(?:please|kindly|can you|could you|would you|show me|give me|tell me|"
    r"i want to see|i would like to see)\s+)+"
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s.%$-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-")
    return _FILLER.sub("", t)


def guard_signature(norm: str, known: set[str]) -> frozenset:
    """Everything that must be IDENTICAL for two requests to share an answer."""
    sig = set()
    for tok in _tokens(norm):
        tok = _NUMWORDS.get(tok, tok)
        if tok in _CANON:
            sig.add(_CANON[tok])
        elif tok.replace(".", "").isdigit():
            sig.add("#" + tok)
        else:
            st = _stem(tok)
            if tok in _STOP or st in _STOP or st in known:
                continue
            sig.add("v:" + st)  # unknown word -> probably a value/entity
    return frozenset(sig)


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
class HashingEmbedder:
    """Dependency-free fallback: hashed unigram + bigram bag of words."""

    dim = 512

    def encode(self, text: str) -> np.ndarray:
        toks = [_stem(t) for t in _tokens(text) if t not in _STOP]
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            v[zlib.crc32(g.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class FastEmbedEmbedder:
    """Small ONNX model, no torch needed:  pip install fastembed"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def encode(self, text: str) -> np.ndarray:
        v = np.asarray(next(iter(self._model.embed([text]))), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v


def make_embedder(kind: str = "auto"):
    if kind in ("auto", "fastembed"):
        try:
            return FastEmbedEmbedder()
        except Exception:
            if kind == "fastembed":
                raise
    return HashingEmbedder()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
@dataclass
class CacheEntry:
    request: str
    norm: str
    ctx: str
    sql: str
    result: str
    sig: frozenset
    vec: np.ndarray
    result_expires: float
    sql_expires: float
    hits: int = 0

    def result_fresh(self) -> bool:
        return time.time() < self.result_expires


class QueryCache:
    def __init__(
        self,
        known_text: str = "",
        max_entries: int = 500,
        result_ttl: float = 300,
        sql_ttl: float = 86_400,
        threshold: float = 0.90,
        embedder: str = "auto",
    ):
        self.max_entries = max_entries
        self.result_ttl = result_ttl
        self.sql_ttl = sql_ttl
        self.threshold = threshold
        self._embedder = make_embedder(embedder)
        # words that are "known" (schema vocabulary) and so are NOT treated as entity values
        self._known = {_stem(w) for w in re.findall(r"[a-z]+", known_text.lower())}
        self._lock = threading.RLock()
        self._entries: "OrderedDict[tuple[str, str], CacheEntry]" = OrderedDict()
        self.stats = {"exact": 0, "semantic": 0, "miss": 0, "stored": 0}
        self.last_miss = ""  # why the last lookup missed (debugging)

    def is_known_noun(self, word: str) -> bool:
        st = _stem(word.lower())
        return len(st) > 3 and st in self._known and st not in _STOP

    # ---- lookup ----------------------------------------------------------
    def lookup(self, request: str, ctx: str = "") -> Optional[tuple[CacheEntry, str, float]]:
        norm = normalize(request)
        with self._lock:
            self._evict_expired()
            key = (norm, ctx)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                entry.hits += 1
                self.stats["exact"] += 1
                return entry, "exact", 1.0
            sig = guard_signature(norm, self._known)
            same_ctx = [e for e in self._entries.values() if e.ctx == ctx]
            cands = [e for e in same_ctx if e.sig == sig]

        if not same_ctx:
            reason = "no cached entries for this user/context"
        elif not cands:
            diff = min((sig ^ e.sig for e in same_ctx), key=len)
            reason = f"guard mismatch, differing tokens: {sorted(diff)}"
        else:  # embed outside the lock
            q = self._embedder.encode(norm)
            sims = np.stack([e.vec for e in cands]) @ q
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                with self._lock:
                    cands[best].hits += 1
                    self.stats["semantic"] += 1
                return cands[best], "semantic", float(sims[best])
            reason = f"best similarity {sims[best]:.3f} < threshold {self.threshold}"

        with self._lock:
            self.stats["miss"] += 1
            self.last_miss = reason
        return None

    # ---- write -----------------------------------------------------------
    def store(self, request: str, ctx: str, sql: str, result: str) -> None:
        norm = normalize(request)
        now = time.time()
        entry = CacheEntry(
            request=request,
            norm=norm,
            ctx=ctx,
            sql=sql,
            result=result,
            sig=guard_signature(norm, self._known),
            vec=self._embedder.encode(norm),
            result_expires=now + self.result_ttl,
            sql_expires=now + self.sql_ttl,
        )
        with self._lock:
            self._entries[(norm, ctx)] = entry
            self._entries.move_to_end((norm, ctx))
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)  # LRU
            self.stats["stored"] += 1

    def refresh(self, entry: CacheEntry, result: str) -> None:
        with self._lock:
            entry.result = result
            entry.result_expires = time.time() + self.result_ttl

    def invalidate(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries.pop((entry.norm, entry.ctx), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_expired(self) -> None:
        now = time.time()
        for k in [k for k, e in self._entries.items() if e.sql_expires < now]:
            del self._entries[k]