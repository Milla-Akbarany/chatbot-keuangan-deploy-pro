"""
app/api/transaction.py
Endpoint untuk akses langsung data transaksi (untuk dashboard/Streamlit).
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional
from app.services.mysql_service import get_transactions, get_total
from app.api.auth import get_current_user
from app.utils.helpers import format_currency

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.get("/list")
def list_transactions(
    period_type: Optional[str] = Query(None, description="month | year | daily"),
    period_value: Optional[str] = Query(None, description="2025-01 | 2025 | 2025-01-15"),
    jenis_akun: Optional[str] = Query(None),
    sub_kategori: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    current_user: dict = Depends(get_current_user),
):
    rows, _ = get_transactions(
        user_id=current_user["user_id"],
        period_type=period_type,
        period_value=period_value,
        jenis_akun=jenis_akun,
        sub_kategori=sub_kategori,
        limit=limit,
    )
    return {"items": rows, "total": len(rows)}


@router.get("/summary")
def summary(
    period_type: Optional[str] = Query(None),
    period_value: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    total_debit, _  = get_total(uid, "debit", period_type, period_value)
    total_kredit, _ = get_total(uid, "kredit", period_type, period_value)
    saldo = total_debit - total_kredit

    return {
        "total_debit":   total_debit,
        "total_kredit":  total_kredit,
        "saldo_bersih":  saldo,
        "formatted": {
            "total_debit":  format_currency(total_debit),
            "total_kredit": format_currency(total_kredit),
            "saldo_bersih": format_currency(saldo),
        },
        "period_type":  period_type,
        "period_value": period_value,
    }
