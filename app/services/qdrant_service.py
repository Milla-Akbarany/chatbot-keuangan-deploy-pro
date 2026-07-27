"""
app/services/qdrant_service.py
Semua operasi ke Qdrant:
- Intent classification (collection: data_intent)
- Entity resolution (collection: dict_user) — SATU panggilan per request
- Upsert transaksi ke collection: data_finance

PERBAIKAN dari prototype lama:
- Tidak ada triple call ke Qdrant untuk satu query
- Tidak ada bypass intent classifier
- UUID sebagai point ID (bukan timestamp — menghindari collision)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue
)
from typing import Optional, Tuple
from dataclasses import dataclass
from app.config.settings import get_settings
import logging
import uuid

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Client Singleton ─────────────────────────────────────────────────────────
_client: Optional[QdrantClient] = None

def get_client() -> QdrantClient:
    global _client

    if _client is None:
        print("URL:", repr(settings.qdrant_url))
        print("API:", repr(settings.qdrant_api_key[:10]))

        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    return _client

# ── Dataclass untuk hasil ────────────────────────────────────────────────────
@dataclass
class IntentResult:
    intent_name: str
    confidence: float
    detected_via: str  # "qdrant" | "unknown"


@dataclass
class EntityResult:
    jenis_akun: str
    sub_kategori: str
    confidence: float
    detected_via: str  # "qdrant" | "rule_based" | "none"


# ── Setup Koleksi ────────────────────────────────────────────────────────────
def ensure_collections():
    """Buat koleksi Qdrant jika belum ada. Dipanggil saat startup."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    for name in [
        settings.collection_intent,
        settings.collection_entity,
        settings.collection_transactions,
    ]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Koleksi Qdrant '{name}' dibuat.")
        else:
            logger.info(f"Koleksi Qdrant '{name}' sudah ada.")


# ── Intent Classification ────────────────────────────────────────────────────
def classify_intent(vector: list[float]) -> IntentResult:
    """
    Deteksi intent dari vektor embedding.
    SEMUA input melalui fungsi ini — tidak ada bypass.
    Return IntentResult dengan intent_name = "unknown" jika confidence < threshold.
    """
    client = get_client()
    results = client.search(
        collection_name=settings.collection_intent,
        query_vector=vector,
        limit=1,
        with_payload=True,
    )

    if not results:
        return IntentResult(intent_name="unknown", confidence=0.0, detected_via="qdrant")

    top = results[0]
    score = top.score

    if score < settings.threshold_intent:
        return IntentResult(intent_name="unknown", confidence=score, detected_via="qdrant")

    intent_name = top.payload.get("intent_name", "unknown")
    return IntentResult(intent_name=intent_name, confidence=score, detected_via="qdrant")


def resolve_entity(vector: list[float], raw_text: str) -> EntityResult:
    """
    Resolusi entitas (jenis_akun + sub_kategori) dalam SATU panggilan Qdrant.
    Jika top result tidak punya sub_kategori, cari hasil berikutnya yang lebih spesifik.
    Jika semua di bawah threshold, fallback ke rule_based.
    """
    client = get_client()
    results = client.search(
        collection_name=settings.collection_entity,
        query_vector=vector,
        limit=5,
        with_payload=True,
    )

    if not results or results[0].score < settings.threshold_entity:
        return _rule_based_entity(raw_text)

    # Cari result terbaik yang punya sub_kategori terisi
    best = results[0]
    if not best.payload.get("sub_kategori"):
        for r in results[1:]:
            if r.score >= settings.threshold_entity and r.payload.get("sub_kategori"):
                best = r
                break

    return EntityResult(
        jenis_akun=best.payload.get("jenis_akun", ""),
        sub_kategori=best.payload.get("sub_kategori", ""),
        confidence=best.score,
        detected_via="qdrant",
    )

def _rule_based_entity(text: str) -> EntityResult:
    """
    Fallback rule-based untuk entity resolution.
    Hanya dipakai jika Qdrant tidak menemukan match di atas threshold.
    Keyword list ini harus diperluas sesuai dengan dict_user.csv.
    """
    text_lower = text.lower()
    rules = [
        # Umum dulu — tanpa sub_kategori
        (["pendapatanku", "pemasukanku", "penghasilanku", "uang masuk"],  "Pendapatan", ""),
        (["pengeluaranku", "uang keluar", "beban"],                   "Beban",      ""),
        # Spesifik — dengan sub_kategori
        (["gajian", "terima gaji", "dapat gaji", "gaji masuk"],     "Pendapatan", "Gaji"),
        (["bayar gaji", "penggajian"],                              "Beban",      "Gaji"),
        (["listrik", "pln", "token listrik"],                       "Beban",      "Listrik"),
        (["makan", "minum", "kopi", "jajan", "snack"],              "Beban",      "Makan/Minum"),
        (["bensin", "bbm", "ojek", "grab", "gojek", "parkir"],      "Beban",      "Transportasi"),
        (["sewa", "kos", "kontrakan"],                              "Beban",      "Sewa"),
        (["pulsa", "internet", "wifi"],                             "Beban",      "Komunikasi"),
        (["belanja", "supermarket", "indomaret", "alfamart"],       "Beban",      "Belanja"),
        (["jual", "penjualan", "hasil jual"],                       "Pendapatan", "Penjualan"),
        (["freelance", "proyek", "fee", "komisi", "bonus"],         "Pendapatan", "Jasa"),
        (["obat", "dokter", "klinik", "apotek", "bpjs"],            "Beban",      "Kesehatan"),
        (["iuran", "sumbangan", "donasi", "sedekah", "zakat"],      "Beban",      "Sosial"),
        (["bimbel", "les", "kursus", "spp", "ukt"],                 "Beban",      "Pendidikan"),
        (["cicilan", "angsuran", "kredit"],                         "Beban",      "Cicilan"),
    ]
    for keywords, jenis, sub in rules:
        if any(k in text_lower for k in keywords):
            return EntityResult(
                jenis_akun=jenis,
                sub_kategori=sub,
                confidence=0.0,
                detected_via="rule_based",
            )

    return EntityResult(
        jenis_akun="",
        sub_kategori="Lain-lain",
        confidence=0.0,
        detected_via="none",
    )


# ── Upsert data ke Qdrant ────────────────────────────────────────────────────
def upsert_intent_sample(vector: list[float], intent_name: str, sample_text: str):
    """Upload satu sample intent ke koleksi data_intent."""
    client = get_client()
    client.upsert(
        collection_name=settings.collection_intent,
        points=[PointStruct(
            id=str(uuid.uuid4()),   # UUID — bukan timestamp
            vector=vector,
            payload={
                "intent_name": intent_name,
                "sample_text": sample_text,
            },
        )],
    )


def upsert_entity_sample(
    vector: list[float],
    keyword: str,
    jenis_akun: str,
    sub_kategori: str,
    sinonim: Optional[str] = None,
):
    """Upload satu entri dict_user ke koleksi dict_user."""
    client = get_client()
    client.upsert(
        collection_name=settings.collection_entity,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "keyword": keyword,
                "jenis_akun": jenis_akun,
                "sub_kategori": sub_kategori,
                "sinonim": sinonim or "",
            },
        )],
    )


def upsert_transaction(
    vector: list[float],
    transaction_id: int,
    user_id: int,
    tanggal: str,
    deskripsi: str,
    jenis_akun: str,
    sub_kategori: str,
    nominal: float,
    tipe: str,  # "debit" | "kredit"
):
    """Upload transaksi ke koleksi data_finance setelah disimpan ke MySQL."""
    client = get_client()
    client.upsert(
        collection_name=settings.collection_transactions,
        points=[PointStruct(
            id=str(uuid.uuid4()),   # UUID — bukan transaction_id (menghindari collision)
            vector=vector,
            payload={
                "transaction_id": transaction_id,
                "user_id": user_id,
                "tanggal": tanggal,
                "deskripsi": deskripsi,
                "jenis_akun": jenis_akun,
                "sub_kategori": sub_kategori,
                "nominal": nominal,
                "tipe": tipe,
            },
        )],
    )


# ── Health check ─────────────────────────────────────────────────────────────
def check_health() -> str:
    try:
        get_client().get_collections()
        return "ok"
    except Exception as e:
        return f"error: {e}"
