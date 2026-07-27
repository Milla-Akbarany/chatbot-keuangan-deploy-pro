"""
app/services/chatbot_service.py
Orchestrator utama pipeline chatbot.

Alur per request:
  preprocess → embed → classify_intent → resolve_entity → parse_temporal
  → route → execute (SQL / template) → [LLM untuk natural response] → log → return response

PERBAIKAN Step 1 - Implementasi LLM yang benar:
  1. LLM dipanggil untuk menghasilkan respons natural dari data database
  2. Fallback ke LLM ketika intent tidak dikenali (unknown)
  3. LLM TIDAK mengarang angka — angka selalu dari MySQL
  4. Jika LLM gagal, fallback ke template statis (aman)
"""

import time
import logging
from datetime import datetime, date
from typing import Optional
from dataclasses import dataclass, field
import traceback

from app.services import embedding_service, qdrant_service, mysql_service, llm_service
from app.utils.helpers import (
    preprocess, parse_temporal, parse_amount,
    extract_description, format_currency, format_transaction_list,
)
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ChatResult:
    log_id: int
    intent: str
    confidence: float
    response: str
    needs_confirmation: bool = False
    pending_data: Optional[dict] = None
    latency_ms: int = 0

def process_message(user_input, session_id, user_id):
    
    pending = _pending_confirmations.get(session_id, {})
    if pending.get("__waiting_clarification") and not pending.get("__waiting_sub_confirm") and not pending.get("__waiting_sub_kategori"):
        return _handle_clarification_answer(user_input, session_id, user_id, pending)
    if pending.get("__waiting_sub_confirm"):
        return _handle_sub_confirm(user_input, session_id, user_id, pending)
    if pending.get("__waiting_sub_kategori"):
        return _handle_sub_input(user_input, session_id, user_id, pending)
# ── Session state (in-memory, untuk pending konfirmasi) ──────────────────────
# Untuk production: ganti dengan Redis
_pending_confirmations: dict[str, dict] = {}


def process_message(
    user_input: str,
    session_id: str,
    user_id: int,
) -> ChatResult:
    """
    Entry point utama untuk setiap pesan user.
    Semua step diukur latency-nya dan dilog ke query_log.
    """
    # ── Cek state klarifikasi sebelum pipeline normal ────────────────────────
    # ── Cek state klarifikasi sebelum pipeline normal ────────────────────────
    pending = _pending_confirmations.get(session_id, {})
    if pending.get("__waiting_clarification") and not pending.get("__waiting_sub_confirm") and not pending.get("__waiting_sub_kategori"):
        return _handle_clarification_answer(user_input, session_id, user_id, pending)
    if pending.get("__waiting_sub_confirm"):
        return _handle_sub_confirm(user_input, session_id, user_id, pending)
    if pending.get("__waiting_sub_kategori"):
        return _handle_sub_input(user_input, session_id, user_id, pending)
    if pending.get("__waiting_delete_id"):
        return _handle_delete_id_answer(user_input, session_id, user_id, pending)

    t_start = time.time()
    ...

    t_start = time.time()
    request_ts = datetime.utcnow()
    log_data: dict = {
        "user_id": user_id,
        "session_id": session_id,
        "request_ts": request_ts,
        "user_input": user_input,
        "intent_threshold_used": settings.threshold_intent,
        "model_version": settings.embedding_model,
        "threshold_version": f"intent={settings.threshold_intent},entity={settings.threshold_entity}",
    }

    try:
        # ── Step 1: Preprocess ───────────────────────────────────────────────
        clean_text = preprocess(user_input)
        log_data["preprocessed_input"] = clean_text

        # ── Step 2: Embed ────────────────────────────────────────────────────
        vector, embed_ms = embedding_service.embed(clean_text)
        log_data["latency_embed_ms"] = embed_ms

        # ── Step 3: Intent Classification (TIDAK ADA bypass) ─────────────────
        t_qdrant = time.time()
        intent_result = qdrant_service.classify_intent(vector)
        intent = intent_result.intent_name
        log_data["predicted_intent"] = intent
        log_data["intent_confidence"] = intent_result.confidence
        log_data["intent_detected_via"] = intent_result.detected_via

        # ── Step 4: Entity Resolution (SATU panggilan Qdrant) ─────────────────
        entity_result = qdrant_service.resolve_entity(vector, clean_text)
        log_data["detected_jenis_akun"] = entity_result.jenis_akun
        log_data["detected_sub_kategori"] = entity_result.sub_kategori
        log_data["entity_confidence"] = entity_result.confidence
        log_data["entity_detected_via"] = entity_result.detected_via
        log_data["latency_qdrant_ms"] = int((time.time() - t_qdrant) * 1000)

        # ── Step 5: Temporal Parsing ──────────────────────────────────────────
        temporal = parse_temporal(clean_text)
        log_data["period_type"] = temporal.period_type
        log_data["period_value"] = temporal.period_value

        # ── Step 6: Route ke handler yang sesuai ─────────────────────────────
        t_sql = time.time()
        result = _route(
            intent=intent,
            clean_text=clean_text,
            vector=vector,
            entity=entity_result,
            temporal=temporal,
            session_id=session_id,
            user_id=user_id,
        )
        log_data["latency_sql_ms"] = int((time.time() - t_sql) * 1000)

        # Update log dengan hasil
        log_data.update({
            "generated_sql": result.get("sql"),
            "sql_success": result.get("sql_success", True),
            "sql_rows_affected": result.get("rows_affected", 0),
            "response_text": result.get("response"),
            "response_type": result.get("response_type", "success"),
        })

    except Exception as e:
        print("=" * 80)
        print("ERROR PROCESS_MESSAGE")
        traceback.print_exc()
        print("=" * 80)

        logger.error(f"Error in process_message: {e}", exc_info=True)

        log_data.update({
            "response_text": "Terjadi kesalahan pada sistem. Silakan coba lagi.",
            "response_type": "system_error",
            "sql_success": False,
        })

        result = {
            "response": log_data["response_text"],
            "needs_confirmation": False,
            "pending_data": None,
            "response_type": "system_error",
        }

    # ── Step 7: Logging (wajib, setiap request) ────────────────────────────
    total_ms = int((time.time() - t_start) * 1000)
    log_data["latency_total_ms"] = total_ms
    log_id = mysql_service.write_query_log(log_data)

    return ChatResult(
        log_id=log_id,
        intent=log_data.get("predicted_intent", "unknown"),
        confidence=log_data.get("intent_confidence", 0.0),
        response=result.get("response", ""),
        needs_confirmation=result.get("needs_confirmation", False),
        pending_data=result.get("pending_data"),
        latency_ms=total_ms,
    )


# ── Konfirmasi transaksi ──────────────────────────────────────────────────────
def process_confirmation(session_id: str, user_id: int, confirm: bool) -> ChatResult:
    """
    Proses konfirmasi user untuk transaksi yang pending.
    Dipanggil setelah user menjawab "ya" atau "tidak".
    """
    pending = _pending_confirmations.pop(session_id, None)
    if not pending:
        return ChatResult(
            log_id=0,
            intent="confirm",
            confidence=1.0,
            response="Tidak ada transaksi yang menunggu konfirmasi.",
            latency_ms=0,
        )
    # ← tambah ini sebelum logika insert yang sudah ada
    if pending.get("__waiting_delete_confirm"):
        if not confirm:
            return ChatResult(
                log_id=0, intent="hapus_transaksi", confidence=1.0,
                response="Baik, transaksi tidak jadi dihapus.",
                latency_ms=0,
            )
        transaction_id = pending["transaction_id"]
        success = mysql_service.delete_transaction(user_id, transaction_id)
        if success:
            return ChatResult(
                log_id=0, intent="hapus_transaksi", confidence=1.0,
                response=f"✅ Transaksi ID {transaction_id} berhasil dihapus.",
                latency_ms=0,
            )
        else:
            return ChatResult(
                log_id=0, intent="hapus_transaksi", confidence=1.0,
                response=f"❌ Gagal menghapus. Transaksi ID {transaction_id} tidak ditemukan.",
                latency_ms=0,
            )

    if not confirm:
        return ChatResult(
            log_id=0,
            intent="confirm",
            confidence=1.0,
            response="Baik, transaksi dibatalkan.",
            latency_ms=0,
        )

    t0 = time.time()
    txn_id = mysql_service.insert_transaction(user_id, pending)

    # Upsert ke Qdrant setelah berhasil tersimpan ke MySQL
    vector, _ = embedding_service.embed(pending.get("deskripsi", ""))
    nominal = pending.get("debit", 0) or pending.get("kredit", 0)
    tipe = "debit" if pending.get("debit", 0) > 0 else "kredit"
    qdrant_service.upsert_transaction(
        vector=vector,
        transaction_id=txn_id,
        user_id=user_id,
        tanggal=str(pending.get("tanggal")),
        deskripsi=pending.get("deskripsi", ""),
        jenis_akun=pending.get("jenis_akun", ""),
        sub_kategori=pending.get("sub_kategori", ""),
        nominal=nominal,
        tipe=tipe,
    )

    latency = int((time.time() - t0) * 1000)
    response = (
        f"✅ Transaksi berhasil dicatat!\n"
        f"📝 {pending.get('deskripsi')}\n"
        f"💰 {format_currency(nominal)}\n"
        f"🏷️ {pending.get('jenis_akun')} → {pending.get('sub_kategori')}\n"
        f"📅 {pending.get('tanggal')}"
    )

    return ChatResult(
        log_id=0,
        intent="confirm",
        confidence=1.0,
        response=response,
        latency_ms=latency,
    )


# ── Router ─────────────────────────────────────────────────────────────────────
def _route(intent, clean_text, vector, entity, temporal, session_id, user_id) -> dict:
    handlers = {
        "catat_transaksi":      _handle_catat,
        "tanya_saldo":          _handle_saldo,
        "tanya_total_akun":     _handle_total,
        "tanya_total_kategori": _handle_total,
        "lihat_rincian":        _handle_rincian,
        "greeting":             _handle_greeting,
        "help":                 _handle_help,
        "hapus_transaksi":      _handle_hapus,
        "unknown":              _handle_unknown,
    }
    handler = handlers.get(intent, _handle_unknown)
    return handler(
        clean_text=clean_text,
        entity=entity,
        temporal=temporal,
        session_id=session_id,
        user_id=user_id,
    )

import traceback
# ── Helper: panggil LLM dengan fallback aman ─────────────────────────────────
def _call_llm_safe(
    user_input: str,
    intent: str,
    context_data: Optional[dict] = None,
    fallback_response: str = "",
) -> str:
    """
    Panggil LLM untuk generate respons natural.
    Jika LLM gagal (error API, timeout, dll), return fallback_response.
    Ini memastikan chatbot TIDAK pernah error karena LLM.
    """
    try:
        llm_response, latency = llm_service.generate_response(
            user_input=user_input,
            intent=intent,
            context_data=context_data,
        )
        if llm_response and llm_response.strip():
            return llm_response
        return fallback_response
    except Exception:
        traceback.print_exc()
        return fallback_response


# ── Handlers ──────────────────────────────────────────────────────────────────
def _handle_catat(clean_text, entity, temporal, session_id, user_id) -> dict:
    """Handler catat transaksi — TIDAK pakai LLM (butuh konfirmasi eksak)."""
    nominal = parse_amount(clean_text)
    if not nominal:
        return {
            "response": "Maaf, saya tidak dapat mendeteksi nominal transaksi. "
                        "Contoh: 'catat beli makan siang 35 ribu'",
            "response_type": "validation_error",
            "sql_success": False,
        }

    deskripsi = extract_description(clean_text)
    tanggal = date.today().isoformat() if temporal.period_type == "none" else temporal.period_value
    
    # PERBAIKAN — tambahkan cek ini sebelum blok if/else:
    if entity.detected_via == "none" or not entity.jenis_akun:
        _pending_confirmations[session_id] = {"__waiting_clarification": True, "nominal": nominal, "deskripsi": deskripsi, "tanggal": tanggal}
        return {
            "response": (
                f"Transaksi '{deskripsi}' senilai {format_currency(nominal)} ini "
                f"termasuk **pemasukan** atau **pengeluaran**?"
            ),
            "needs_confirmation": False,   # bukan konfirmasi simpan, tapi klarifikasi jenis
            "response_type": "clarification",
            "sql_success": True,
        }

    # Tentukan debit/kredit berdasarkan jenis akun
    jenis = entity.jenis_akun.lower()
    if "pendapatan" in jenis or "aset" in jenis:
        debit, kredit = nominal, 0
    else:
        debit, kredit = 0, nominal

    pending_data = {
        "tanggal":     tanggal,
        "deskripsi":   deskripsi,
        "debit":       debit,
        "kredit":      kredit,
        "jenis_akun":  entity.jenis_akun,
        "sub_kategori": entity.sub_kategori,
    }
    _pending_confirmations[session_id] = pending_data

    tipe_str = "pemasukan" if debit > 0 else "pengeluaran"
    response = (
        f"Konfirmasi pencatatan {tipe_str}:\n"
        f"📝 Deskripsi  : {deskripsi}\n"
        f"💰 Nominal    : {format_currency(nominal)}\n"
        f"🏷️ Kategori   : {entity.jenis_akun} → {entity.sub_kategori}\n"
        f"📅 Tanggal    : {tanggal}\n\n"
        f"Ketik 'ya' untuk menyimpan atau 'batal' untuk membatalkan."
    )
    return {
        "response": response,
        "needs_confirmation": True,
        "pending_data": pending_data,
        "response_type": "success",
        "sql_success": True,
        "rows_affected": 0,
    }


def _handle_saldo(clean_text, entity, temporal, session_id, user_id) -> dict:
    """Handler tanya saldo — data dari MySQL, respons natural dari LLM."""
    total_debit, sql_d = mysql_service.get_total(
        user_id, "debit", temporal.period_type, temporal.period_value)
    total_kredit, sql_k = mysql_service.get_total(
        user_id, "kredit", temporal.period_type, temporal.period_value)
    saldo = total_debit - total_kredit

    period_str = temporal.period_value if temporal.period_value else "Semua waktu"

    # Template fallback (jika LLM gagal)
    fallback = (
        f"💰 Saldo Keuangan ({period_str})\n"
        f"Total Pemasukan : {format_currency(total_debit)}\n"
        f"Total Pengeluaran: {format_currency(total_kredit)}\n"
        f"Saldo Bersih    : {format_currency(saldo)}"
    )

    # Panggil LLM untuk respons natural
    context_data = {
        "total_debit": format_currency(total_debit),
        "total_kredit": format_currency(total_kredit),
        "saldo": format_currency(saldo),
        "periode": period_str,
    }
    response = _call_llm_safe(
        user_input=clean_text,
        intent="tanya_saldo",
        context_data=context_data,
        fallback_response=fallback,
    )

    return {
        "response": response,
        "sql": f"{sql_d} | {sql_k}",
        "sql_success": True,
        "rows_affected": 1,
        "response_type": "success",
    }


def _handle_total(clean_text, entity, temporal, session_id, user_id) -> dict:
    """Handler tanya total per akun/kategori — data dari MySQL, respons dari LLM."""
    # Tentukan kolom berdasarkan jenis akun
    # Pendapatan & Aset → debit (pemasukan)
    # Beban & Kewajiban → kredit (pengeluaran)
    jenis_lower = entity.jenis_akun.lower() if entity.jenis_akun else ""
    if "pendapatan" in jenis_lower or "aset" in jenis_lower:
        kolom = "debit"
    else:
        kolom = "kredit"

    # Hanya filter by entity jika benar-benar terdeteksi (bukan default fallback)
    # detected_via="none" artinya tidak ada keyword spesifik yang ditemukan
    # contoh: "pengeluaran minggu ini" -> jangan filter sub_kategori
    # contoh: "total makan minggu ini" -> filter sub_kategori=Makan/Minum
    use_jenis = entity.jenis_akun if entity.jenis_akun else None
    use_sub = entity.sub_kategori if entity.sub_kategori else None

    total, sql = mysql_service.get_total(
        user_id, kolom,
        temporal.period_type, temporal.period_value,
        use_jenis,
        use_sub,
    )

    period_str = temporal.period_value if temporal.period_value else "Semua waktu"
    sub_str = f" ({entity.sub_kategori})" if entity.sub_kategori and entity.detected_via != "none" else ""

    # Template fallback
    fallback = f"Total{sub_str} {period_str}: {format_currency(total)}"

    # Panggil LLM untuk respons natural
    context_data = {
        "total": format_currency(total),
        "jenis_akun": entity.jenis_akun or "-",
        "sub_kategori": entity.sub_kategori or "-",
        "periode": period_str,
    }
    response = _call_llm_safe(
        user_input=clean_text,
        intent="tanya_total_kategori",
        context_data=context_data,
        fallback_response=fallback,
    )

    return {
        "response": response,
        "sql": sql,
        "sql_success": True,
        "rows_affected": 1,
        "response_type": "success",
    }


def _handle_rincian(clean_text, entity, temporal, session_id, user_id) -> dict:
    """Handler lihat rincian transaksi — data dari MySQL, ringkasan dari LLM."""
    # Hanya filter by entity jika benar-benar terdeteksi (bukan default fallback)
    use_jenis = entity.jenis_akun if entity.jenis_akun else None
    use_sub = entity.sub_kategori if (entity.sub_kategori and entity.detected_via == "qdrant" and entity.confidence > 0.85) else None

    rows, sql = mysql_service.get_transactions(
        user_id,
        temporal.period_type, temporal.period_value,
        use_jenis,
        use_sub,
    )

    formatted_list = format_transaction_list(rows)
    period_str = temporal.period_value if temporal.period_value else "Semua waktu"
    header = f"📋 Rincian Transaksi ({period_str}):\n"

    # Template fallback
    fallback = header + formatted_list

    # Panggil LLM untuk ringkasan natural
    context_data = {
        "transaksi": rows,
        "transaksi_formatted": formatted_list,
        "jumlah_transaksi": len(rows),
        "periode": period_str,
    }
    response = _call_llm_safe(
        user_input=clean_text,
        intent="lihat_rincian",
        context_data=context_data,
        fallback_response=fallback,
    )

    return {
        "response": response,
        "sql": sql,
        "sql_success": True,
        "rows_affected": len(rows),
        "response_type": "success",
    }


def _handle_greeting(clean_text, *args, **kwargs) -> dict:
    """Handler greeting — LLM buat sapaan natural, fallback ke template."""
    fallback = (
        "Halo! Saya asisten keuangan Anda 👋\n"
        "Saya bisa membantu Anda:\n"
        "• Mencatat transaksi (ketik 'catat ...')\n"
        "• Cek saldo (ketik 'cek saldo')\n"
        "• Lihat rincian transaksi\n"
        "• Hitung total pengeluaran/pemasukan\n\n"
        "Apa yang bisa saya bantu?"
    )

    response = _call_llm_safe(
        user_input=clean_text,
        intent="greeting",
        fallback_response=fallback,
    )

    return {
        "response": response,
        "response_type": "success",
        "sql_success": True,
        "rows_affected": 0,
    }


def _handle_help(*args, **kwargs) -> dict:
    """Handler help — template statis (tidak perlu LLM untuk panduan)."""
    return {
        "response": (
            "📖 Panduan Penggunaan:\n\n"
            "🔹 Catat pengeluaran: 'catat beli makan 35 ribu'\n"
            "🔹 Catat pemasukan : 'catat gajian 5 juta'\n"
            "🔹 Cek saldo       : 'cek saldo bulan ini'\n"
            "🔹 Total kategori  : 'total makan bulan januari'\n"
            "🔹 Lihat rincian   : 'lihat transaksi bulan ini'\n"
        ),
        "response_type": "success",
        "sql_success": True,
        "rows_affected": 0,
    }


def _handle_unknown(clean_text, *args, **kwargs) -> dict:
    """
    Handler unknown — FALLBACK KE LLM.
    Ini adalah perbaikan utama: ketika intent tidak dikenali,
    LLM akan mencoba menjawab secara natural sebagai asisten keuangan.
    """
    fallback = (
        "Maaf, saya kurang memahami permintaan Anda. "
        "Ketik 'bantuan' untuk melihat contoh perintah yang tersedia."
    )

    response = _call_llm_safe(
        user_input=clean_text,
        intent="unknown",
        fallback_response=fallback,
    )

    return {
        "response": response,
        "response_type": "low_confidence",
        "sql_success": True,
        "rows_affected": 0,
    }

def _suggest_sub_kategori(deskripsi: str, jenis_akun: str) -> str:
    """Tebak sub_kategori dari deskripsi untuk ditampilkan sebagai saran."""
    text = deskripsi.lower()
    
    if jenis_akun == "Beban":
        rules = [
            (["bensin", "bbm", "ojek", "grab", "gojek", "parkir", "tol", "motor", "mobil"], "Transportasi"),
            (["makan", "minum", "kopi", "warung", "resto", "jajan", "snack"],               "Makan/Minum"),
            (["listrik", "pln", "token"],                                                    "Listrik"),
            (["wifi", "internet", "pulsa", "data"],                                          "Komunikasi"),
            (["sewa", "kos", "kontrakan"],                                                   "Sewa"),
            (["obat", "dokter", "klinik", "apotek", "bpjs"],                                "Kesehatan"),
            (["belanja", "supermarket", "indomaret", "alfamart"],                            "Belanja"),
            (["iuran", "sumbangan", "donasi", "sedekah", "zakat"],                          "Sosial"),
            (["laptop", "hp", "gadget", "elektronik"],                                       "Peralatan"),
            (["cicilan", "angsuran", "kredit"],                                              "Cicilan"),
        ]
    else:  # Pendapatan
        rules = [
            (["gaji", "honor", "upah", "salary"],          "Gaji"),
            (["freelance", "proyek", "fee", "komisi"],      "Jasa"),
            (["jual", "penjualan"],                         "Penjualan"),
            (["bonus"],                                     "Bonus"),
            (["bunga", "investasi", "dividen"],             "Investasi"),
        ]
    
    for keywords, sub in rules:
        if any(k in text for k in keywords):
            return sub
    return "Lain-lain"

def _handle_clarification_answer(user_input, session_id, user_id, pending) -> ChatResult:
    text = user_input.lower().strip()

    # Cek batal dulu
    is_batal = any(k in text for k in ["tidak", "batal", "cancel", "gak jadi", "ga jadi", "nggak"])
    if is_batal:
        _pending_confirmations.pop(session_id, None)
        return ChatResult(
            log_id=0, intent="clarification", confidence=1.0,
            response="Baik, pencatatan dibatalkan.",
            latency_ms=0,
        )

    is_pemasukan = any(k in text for k in ["pemasukan", "masuk", "pendapatan", "terima", "dapat"])
    is_pengeluaran = any(k in text for k in ["pengeluaran", "keluar", "beban", "bayar", "beli"])

    if not is_pemasukan and not is_pengeluaran:
        return ChatResult(
            log_id=0, intent="clarification", confidence=1.0,
            response="Maaf, belum paham. Ini pemasukan (uang masuk) atau pengeluaran (uang keluar)? Atau ketik 'batal' untuk membatalkan.",
            latency_ms=0,
        )

    jenis_akun = "Pendapatan" if is_pemasukan else "Beban"

    # Tebak sub kategori dari deskripsi
    sub_saran = _suggest_sub_kategori(pending.get("deskripsi", ""), jenis_akun)

    if sub_saran == "Lain-lain":
        # Tidak bisa tebak, tanya ke user
        _pending_confirmations[session_id] = {
            **pending,
            "__waiting_sub_kategori": True,
            "__waiting_clarification": False,
            "jenis_akun": jenis_akun,
        }
        return ChatResult(
            log_id=0, intent="clarification", confidence=1.0,
            response=(
                "Masuk kategori apa ini?\n"
                "Contoh: Transportasi, Makan/Minum, Belanja, Kesehatan, Komunikasi, Sosial, Lain-lain"
            ),
            latency_ms=0,
        )
    
    else:
        # Ada saran, konfirmasi dulu ke user
        _pending_confirmations[session_id] = {
            **pending,
            "__waiting_clarification": False,
            "__waiting_sub_confirm": True,
            "jenis_akun": jenis_akun,
            "sub_kategori": sub_saran,
        }
        return ChatResult(
            log_id=0, intent="clarification", confidence=1.0,
            response=f"Masuk kategori {sub_saran}? Ketik 'ya' atau sebutkan kategori lain.",
            latency_ms=0,
        )

def _handle_sub_confirm(user_input, session_id, user_id, pending) -> ChatResult:
    """User mengkonfirmasi atau mengganti saran sub kategori."""
    text = user_input.lower().strip()

    is_batal = any(k in text for k in ["tidak", "batal", "cancel", "gak jadi", "ga jadi", "nggak"])
    if is_batal:
        _pending_confirmations.pop(session_id, None)
        return ChatResult(
            log_id=0, intent="clarification", confidence=1.0,
            response="Baik, pencatatan dibatalkan.",
            latency_ms=0,
        )

    if any(k in text for k in ["ya", "iya", "yep", "ok", "oke", "benar", "betul"]):
        # Pakai saran yang sudah ada
        sub = pending["sub_kategori"]
    else:
        # User sebut kategori lain, pakai input user langsung
        sub = user_input.strip().capitalize()

    return _finalize_clarification(session_id, pending, pending["jenis_akun"], sub)


def _handle_sub_input(user_input, session_id, user_id, pending) -> ChatResult:
    """User mengetik kategori secara manual."""
    text = user_input.lower().strip()

    is_batal = any(k in text for k in ["tidak", "batal", "cancel", "gak jadi", "ga jadi", "nggak"])
    if is_batal:
        _pending_confirmations.pop(session_id, None)
        return ChatResult(
            log_id=0, intent="clarification", confidence=1.0,
            response="Baik, pencatatan dibatalkan.",
            latency_ms=0,
        )

    sub = user_input.strip().capitalize()
    return _finalize_clarification(session_id, pending, pending["jenis_akun"], sub)


def _finalize_clarification(session_id, pending, jenis_akun, sub_kategori) -> ChatResult:
    """Setelah jenis dan sub diketahui, lanjut ke konfirmasi simpan."""
    nominal = pending["nominal"]
    debit  = nominal if jenis_akun == "Pendapatan" else 0
    kredit = 0 if debit > 0 else nominal

    final_pending = {
        "tanggal":      pending["tanggal"],
        "deskripsi":    pending["deskripsi"],
        "debit":        debit,
        "kredit":       kredit,
        "jenis_akun":   jenis_akun,
        "sub_kategori": sub_kategori,
    }
    _pending_confirmations[session_id] = final_pending

    tipe_str = "pemasukan" if debit > 0 else "pengeluaran"

    return ChatResult(
        log_id=0, intent="clarification", confidence=1.0,
        response=(
            f"Konfirmasi pencatatan {tipe_str}:\n"
            f"📝 Deskripsi  : {final_pending['deskripsi']}\n"
            f"💰 Nominal    : {format_currency(nominal)}\n"
            f"🏷️ Kategori   : {jenis_akun} → {sub_kategori}\n"
            f"📅 Tanggal    : {final_pending['tanggal']}\n\n"
            f"Ketik 'ya' untuk menyimpan atau 'batal' untuk membatalkan."
        ),
        needs_confirmation=True,
        latency_ms=0,
    )

def _handle_hapus(clean_text, entity, temporal, session_id, user_id) -> dict:
    """Tampilkan transaksi terbaru, minta user pilih ID yang mau dihapus."""
    rows = mysql_service.get_recent_transactions(user_id, limit=5)
    
    if not rows:
        return {
            "response": "Tidak ada transaksi yang bisa dihapus.",
            "response_type": "success",
            "sql_success": True,
            "rows_affected": 0,
        }
    
    # Simpan daftar ID yang valid ke pending, supaya user tidak bisa input ID sembarangan
    valid_ids = [row["id"] for row in rows]
    _pending_confirmations[session_id] = {
        "__waiting_delete_id": True,
        "valid_ids": valid_ids,
    }
    
    lines = ["Pilih transaksi yang ingin dihapus (ketik nomornya):"]
    for row in rows:
        nominal = float(row.get("debit") or 0) or float(row.get("kredit") or 0)
        tipe = "(+)" if float(row.get("debit") or 0) > 0 else "(-)"
        lines.append(
            f"ID {row['id']} | {str(row['tanggal'])[:10]} | "
            f"{row['deskripsi']} | {tipe} {format_currency(nominal)}"
        )
    lines.append("\nContoh: ketik '42' untuk hapus transaksi ID 42")
    
    return {
        "response": "\n".join(lines),
        "needs_confirmation": False,
        "response_type": "success",
        "sql_success": True,
        "rows_affected": 0,
    }

def _handle_delete_id_answer(user_input, session_id, user_id, pending) -> ChatResult:
    """Proses jawaban user setelah ditampilkan daftar transaksi untuk dihapus."""
        # Tambahkan cek batal di paling atas
    text = user_input.lower().strip() 
    is_batal = any(k in text for k in ["tidak", "batal", "cancel", "gak", "ga", "nggak"])
    if is_batal:
        _pending_confirmations.pop(session_id, None)
        return ChatResult(
            log_id=0, intent="hapus_transaksi", confidence=1.0,
            response="Baik, tidak jadi menghapus transaksi.",
            latency_ms=0,
        )
    import re
    m = re.search(r"\d+", user_input.strip())
    
    if not m:
        return ChatResult(
            log_id=0, intent="hapus_transaksi", confidence=1.0,
            response="Ketik ID transaksi yang ingin dihapus (angka saja). Contoh: '42'",
            latency_ms=0,
        )
    
    transaction_id = int(m.group())
    valid_ids = pending.get("valid_ids", [])
    
    # Validasi — ID harus ada di daftar yang ditampilkan
    if transaction_id not in valid_ids:
        return ChatResult(
            log_id=0, intent="hapus_transaksi", confidence=1.0,
            response=f"ID {transaction_id} tidak ada di daftar. Pilih ID yang tertera di atas.",
            latency_ms=0,
        )
    
    # Simpan ke pending konfirmasi hapus
    _pending_confirmations[session_id] = {
        "__waiting_delete_confirm": True,
        "transaction_id": transaction_id,
    }
    
    return ChatResult(
        log_id=0, intent="hapus_transaksi", confidence=1.0,
        response=f"Yakin ingin menghapus transaksi ID {transaction_id}? Ketik 'ya' untuk konfirmasi atau 'batal'.",
        needs_confirmation=True,
        latency_ms=0,
    )