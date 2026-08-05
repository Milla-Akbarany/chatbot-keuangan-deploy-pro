"""
streamlit_ui/app.py
UI chatbot berbasis Streamlit dengan Dashboard Analitik.

Halaman:
  - 💬 Chat   : percakapan dengan chatbot keuangan
  - 📊 Dashboard : ringkasan & grafik keuangan per user

Jalankan: streamlit run streamlit_ui/app.py
"""

import streamlit as st
import requests
import uuid
import pandas as pd
from datetime import datetime
import os

os.environ["STREAMLIT_USE_ARROW"] = "0"

# ── Konfigurasi ───────────────────────────────────────────────────────────────
API_BASE = "https://financial-chatbot-backend-production.up.railway.app"

st.set_page_config(
    page_title="Chatbot Keuangan",
    page_icon="💰",
    layout="wide",
)


# ── Helper API calls ──────────────────────────────────────────────────────────
def api_post(endpoint: str, data: dict, token: str = None):

    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:

        r = requests.post(
            f"{API_BASE}{endpoint}",
            json=data,
            headers=headers,
            timeout=30,
        )

        print(r.status_code)
        print(r.text)

        return r.json()

    except Exception as e:

        print(e)

        return {
            "error": str(e)
        }


def api_get(endpoint: str, params: dict = None, token: str = None) -> dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, headers=headers, timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fmt_rp(amount: float) -> str:
    """Format angka ke Rupiah Indonesia."""
    return f"Rp {amount:,.0f}".replace(",", ".")


# ── Session State Init ────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_confirm" not in st.session_state:
    st.session_state.pending_confirm = False
if "page" not in st.session_state:
    st.session_state.page = "chat"


# ── Login / Register ──────────────────────────────────────────────────────────
def show_login():
    st.title("💰 Chatbot Keuangan")
    tab_login, tab_register = st.tabs(["Login", "Daftar"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Masuk", type="primary"):
            r = requests.post(
                f"{API_BASE}/auth/login",
                data={"username": username, "password": password},
                timeout=10,
            )

            st.write("Status:", r.status_code)
            st.write("Response:", r.text)

            if r.headers.get("content-type", "").startswith("application/json"):
                resp = r.json()
                if "access_token" in resp:
                    st.session_state.token = resp["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(resp.get("detail", "Login gagal."))
            else:
                st.error("Backend tidak mengembalikan JSON.")
        
    with tab_register:
        new_user = st.text_input("Username baru", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")
        full_name = st.text_input("Nama lengkap (opsional)", key="reg_name")
        if st.button("Daftar", type="primary"):
            resp = api_post("/auth/register", {
                "username": new_user,
                "password": new_pass,
                "full_name": full_name,
            })
            if "user_id" in resp:
                st.success("Registrasi berhasil! Silakan login.")
            else:
                st.error(resp.get("detail", "Registrasi gagal."))


# ── Chat Page ─────────────────────────────────────────────────────────────────
def show_chat():
    st.title(f"💰 Chatbot Keuangan — {st.session_state.username}")

    col_chat, col_dashboard = st.columns([2, 1])

    with col_chat:
        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg.get("latency"):
                        st.caption(f"⏱ {msg['latency']}ms")

        if st.session_state.pending_confirm:
            col_ya, col_batal = st.columns(2)
            with col_ya:
                if st.button("✅ Ya, simpan", type="primary", use_container_width=True):
                    _send_confirmation(True)
            with col_batal:
                if st.button("❌ Batal", use_container_width=True):
                    _send_confirmation(False)
        else:
            prompt = st.chat_input("Ketik pesan Anda...")
            if prompt:
                _send_message(prompt)

        if st.button("🔄 Mulai percakapan baru"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.pending_confirm = False
            st.rerun()

    with col_dashboard:
        _show_mini_dashboard()


def _send_message(text: str):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": text,
        }
    )

    resp = api_post(
        "/chat/message",
        {
            "session_id": st.session_state.session_id,
            "message": text,
        },
        token=st.session_state.token,
    )
    st.write(resp)

    print(resp)

    if "error" in resp:
        st.error(resp["error"])

    st.write(resp)

    bot_msg = resp.get("response", "Terjadi kesalahan.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_msg,
            "latency": resp.get("latency_ms"),
        }
    )

    if resp.get("needs_confirmation"):
        st.session_state.pending_confirm = True

    st.rerun()

def _send_confirmation(confirm: bool):
    st.session_state.pending_confirm = False
    resp = api_post(
        "/chat/confirm",
        {"session_id": st.session_state.session_id, "confirm": confirm},
        token=st.session_state.token,
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp.get("response", ""),
    })
    st.rerun()


def _show_mini_dashboard():
    """Mini dashboard di samping chat — ringkasan bulan ini."""
    now = datetime.now()
    period = now.strftime("%Y-%m")
    data = api_get(
        "/transactions/summary",
        params={"period_type": "month", "period_value": period},
        token=st.session_state.token,
    )

    st.subheader("📊 Ringkasan Bulan Ini")
    if "formatted" in data:
        f = data["formatted"]
        saldo = data.get("saldo_bersih", 0)
        st.metric("Pemasukan", f.get("total_debit", "-"))
        st.metric("Pengeluaran", f.get("total_kredit", "-"))
        st.metric(
            "Saldo Bersih",
            f.get("saldo_bersih", "-"),
            delta="Surplus" if saldo >= 0 else "Defisit",
            delta_color="normal" if saldo >= 0 else "inverse",
        )
    else:
        st.info("Data belum tersedia.")

    st.divider()
    st.subheader("📋 Transaksi Terakhir")
    txn_data = api_get(
        "/transactions/list",
        params={"period_type": "month", "period_value": period, "limit": 5},
        token=st.session_state.token,
    )
    items = txn_data.get("items", [])
    if items:
        for item in items:
            debit = float(item.get("debit", 0) or 0)
            kredit = float(item.get("kredit", 0) or 0)
            nominal = debit if debit > 0 else kredit
            tipe = "⬆️" if debit > 0 else "⬇️"
            st.write(f"{tipe} **{item.get('sub_kategori', '-')}** — {fmt_rp(nominal)}")
            st.caption(str(item.get("tanggal", ""))[:10])
    else:
        st.info("Belum ada transaksi bulan ini.")


# ── Dashboard Page ────────────────────────────────────────────────────────────

def show_dashboard():
    st.title(f"📊 Dashboard Keuangan — {st.session_state.username}")

    now = datetime.now()

    # ── Filter Periode ────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        period_option = st.selectbox(
            "Pilih Periode",
            ["Bulan Ini", "Bulan Lalu", "3 Bulan Terakhir", "6 Bulan Terakhir", "Tahun Ini"],
        )

    # Map pilihan ke API params
    if period_option == "Bulan Ini":
        period_type = "month"
        period_value = now.strftime("%Y-%m")
        period_label = now.strftime("Bulan %B %Y")
    elif period_option == "Bulan Lalu":
        m = now.month - 1 if now.month > 1 else 12
        y = now.year if now.month > 1 else now.year - 1
        period_type = "month"
        period_value = f"{y}-{m:02d}"
        period_label = f"Bulan {m}/{y}"
    elif period_option == "3 Bulan Terakhir":
        m, y = now.month - 3, now.year
        while m <= 0:
            m += 12
            y -= 1
        period_type = "range"
        period_value = f"{y}-{m:02d}-01:{now.strftime('%Y-%m-%d')}"
        period_label = "3 Bulan Terakhir"
    elif period_option == "6 Bulan Terakhir":
        m, y = now.month - 6, now.year
        while m <= 0:
            m += 12
            y -= 1
        period_type = "range"
        period_value = f"{y}-{m:02d}-01:{now.strftime('%Y-%m-%d')}"
        period_label = "6 Bulan Terakhir"
    else:  # Tahun Ini
        period_type = "year"
        period_value = str(now.year)
        period_label = f"Tahun {now.year}"

    # ── Ambil data summary ──────────────────────────────────────────────────
    summary = api_get(
        "/transactions/summary",
        params={"period_type": period_type, "period_value": period_value},
        token=st.session_state.token,
    )
    if "error" in summary:
        st.error(f"Gagal mengambil ringkasan: {summary['error']}")
        return

    total_debit = summary.get("total_debit", 0)
    total_kredit = summary.get("total_kredit", 0)
    saldo = summary.get("saldo_bersih", 0)

    # ── Ambil data list transaksi ───────────────────────────────────────────
    txn_data = api_get(
        "/transactions/list",
        params={"period_type": period_type, "period_value": period_value, "limit": 500},
        token=st.session_state.token,
    )
    if "error" in txn_data:
        st.error(f"Gagal mengambil daftar transaksi: {txn_data['error']}")
        items = []
    else:
        items = txn_data.get("items", [])

    jumlah_transaksi = len(items)
    st.write("DEBUG jumlah transaksi:", len(items))
    st.write("DEBUG sample item:", items[:1])

    # ── 1. KPI Cards ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Pemasukan", fmt_rp(total_debit))
    col2.metric("💸 Total Pengeluaran", fmt_rp(total_kredit))
    col3.metric(
        "📈 Saldo Bersih",
        fmt_rp(saldo),
        delta="Surplus" if saldo >= 0 else "Defisit",
        delta_color="normal" if saldo >= 0 else "inverse",
    )
    col4.metric("📋 Jumlah Transaksi", jumlah_transaksi)

    st.divider()

    # ── 2. Tren 6 Bulan Terakhir (line chart) ──────────────────────────────
    st.subheader("📈 Tren 6 Bulan Terakhir")
    trend_rows = []
    for i in range(5, -1, -1):
        m, y = now.month - i, now.year
        while m <= 0:
            m += 12
            y -= 1
        pv = f"{y}-{m:02d}"
        d = api_get(
            "/transactions/summary",
            params={"period_type": "month", "period_value": pv},
            token=st.session_state.token,
        )
        if "error" not in d:
            trend_rows.append({
                "Bulan": f"{m:02d}/{y}",
                "Pemasukan": round(d.get("total_debit", 0), 0),
                "Pengeluaran": round(d.get("total_kredit", 0), 0),
            })

    if trend_rows:
        df_trend = pd.DataFrame(trend_rows).set_index("Bulan")
        st.line_chart(df_trend)
    else:
        st.info("Belum ada data transaksi 6 bulan terakhir.")

    st.divider()

    # ── 3 & 4. Pengeluaran & Pemasukan per Kategori (bar chart) ─────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"🏷️ Pengeluaran per Kategori ({period_label})")
        if items:
            cat_kredit = {}
            for item in items:
                kredit = float(item.get("kredit", 0) or 0)
                if kredit > 0:
                    cat = item.get("sub_kategori") or "Lain-lain"
                    cat_kredit[cat] = cat_kredit.get(cat, 0) + kredit
            if cat_kredit:
                df_kredit = (
                    pd.DataFrame({"Kategori": list(cat_kredit.keys()), "Total": list(cat_kredit.values())})
                    .set_index("Kategori")
                    .sort_values("Total", ascending=False)
                )
                st.bar_chart(df_kredit)
            else:
                st.info("Belum ada pengeluaran pada periode ini.")
        else:
            st.info("Belum ada transaksi pada periode ini.")

    with col_right:
        st.subheader(f"💰 Pemasukan per Kategori ({period_label})")
        if items:
            cat_debit = {}
            for item in items:
                debit = float(item.get("debit", 0) or 0)
                if debit > 0:
                    cat = item.get("sub_kategori") or "Lain-lain"
                    cat_debit[cat] = cat_debit.get(cat, 0) + debit
            if cat_debit:
                df_debit = (
                    pd.DataFrame({"Kategori": list(cat_debit.keys()), "Total": list(cat_debit.values())})
                    .set_index("Kategori")
                    .sort_values("Total", ascending=False)
                )
                st.bar_chart(df_debit)
            else:
                st.info("Belum ada pemasukan pada periode ini.")
        else:
            st.info("Belum ada transaksi pada periode ini.")

    st.divider()

    # ── 5. Komposisi Keuangan (bar chart) ──────────────────────────────────
    st.subheader(f"📊 Komposisi Keuangan ({period_label})")
    if total_debit > 0 or total_kredit > 0:
        df_komposisi = pd.DataFrame({
            "Kategori": ["Pemasukan", "Pengeluaran"],
            "Total": [total_debit, total_kredit],
        }).set_index("Kategori")
        st.bar_chart(df_komposisi)
    else:
        st.info("Belum ada data keuangan pada periode ini.")

    st.divider()

    # ── 6. Transaksi Terbaru (5 terakhir) ──────────────────────────────────
    st.subheader(f"📋 Transaksi Terbaru ({period_label})")
    if items:
        # Asumsi items sudah diurutkan dari API (terbaru pertama)
        latest = items[:5]
        rows = []
        for item in latest:
            debit = float(item.get("debit", 0) or 0)
            kredit = float(item.get("kredit", 0) or 0)
            nominal = debit if debit > 0 else kredit
            tipe = "⬆️ Masuk" if debit > 0 else "⬇️ Keluar"
            rows.append({
                "Tanggal": str(item.get("tanggal", ""))[:10],
                "Deskripsi": item.get("deskripsi", "-"),
                "Kategori": item.get("sub_kategori", "-"),
                "Jenis Akun": item.get("jenis_akun", "-"),
                "Tipe": tipe,
                "Nominal": fmt_rp(nominal),
            })
        df_latest = pd.DataFrame(rows)
        st.dataframe(df_latest, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada transaksi pada periode ini.")

    st.divider()

    # ── 7. Download CSV ─────────────────────────────────────────────────────
    if items:
        all_rows = []
        for item in items:
            debit = float(item.get("debit", 0) or 0)
            kredit = float(item.get("kredit", 0) or 0)
            nominal = debit if debit > 0 else kredit
            tipe = "Masuk" if debit > 0 else "Keluar"
            all_rows.append({
                "Tanggal": str(item.get("tanggal", ""))[:10],
                "Deskripsi": item.get("deskripsi", "-"),
                "Kategori": item.get("sub_kategori", "-"),
                "Jenis Akun": item.get("jenis_akun", "-"),
                "Tipe": tipe,
                "Nominal": nominal,
            })
        df_all = pd.DataFrame(all_rows)
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"transaksi_{period_value}.csv",
            mime="text/csv",
        )
        
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.token:
        show_login()
        return

    # Sidebar navigasi
    with st.sidebar:
        st.write(f"👤 **{st.session_state.username}**")
        st.divider()

        menu = st.radio(
            "Menu",
            ["💬 Chat", "📊 Dashboard"],
            label_visibility="collapsed",
        )

        st.divider()

        if st.button("🚪 Keluar", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.messages = []
            st.session_state.pending_confirm = False
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # Halaman
    if menu == "💬 Chat":
        show_chat()
    elif menu == "📊 Dashboard":
        show_dashboard()


if __name__ == "__main__":
    main()