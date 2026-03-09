import streamlit as st
import pandas as pd
import ta
import platform
import matplotlib.pyplot as plt
from datetime import timedelta
import yfinance as yf
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import requests
from bs4 import BeautifulSoup
import stock_eng_to_cn





# ------------------------
# 判斷平台，設定中文字型
if platform.system() == "Windows":
    CHINESE_FONT = 'Microsoft JhengHei'
elif platform.system() == "Darwin":  # Mac
    CHINESE_FONT = 'Heiti TC'  # 或 'PingFang TC'
else:
    CHINESE_FONT = 'DejaVu Sans'  # Linux / fallback

plt.rcParams['font.sans-serif'] = [CHINESE_FONT]
plt.rcParams['axes.unicode_minus'] = False


# ===============================
# Sidebar｜期末專案 - 時間序列分析
# ===============================
st.sidebar.title("📊 期末專案 - 時間序列分析")
st.sidebar.markdown("**G140A006 葉天明**")
st.sidebar.markdown("---")

# ===============================
# Sidebar｜資料來源（全域）
# ===============================
st.sidebar.markdown("### 📂 資料來源")

data_source = st.sidebar.selectbox(
    "",
    ["Yahoo Finance", "CSV 上傳"]
)
# ===============================
# Sidebar｜股票分類資料
# ===============================
stock_sector_map = {
    "Semiconductors (Tech)": {
        "2330.TW": "TSMC",
        "2454.TW": "MediaTek",
        "2303.TW": "UMC",
        "3711.TW": "ASE Technology",
        "6488.TWO": "GlobalWafers"
    },
    "AI & Computer Hardware": {
        "2317.TW": "Hon Hai (Foxconn)",
        "2382.TW": "Quanta Computer",
        "6669.TW": "Wiwynn",
        "3231.TW": "Wistron",
        "2357.TW": "ASUSTeK"
    },
    "Financials & Banking": {
        "2881.TW": "Fubon Financial",
        "2882.TW": "Cathay Financial",
        "2891.TW": "CTBC Financial",
        "2886.TW": "Mega Financial",
        "2884.TW": "E.SUN Financial"
    },
    "Shipping & Transportation": {
        "2603.TW": "Evergreen Marine",
        "2609.TW": "Yang Ming Marine",
        "2615.TW": "Wan Hai Lines",
        "2618.TW": "EVA Airways",
        "2610.TW": "China Airlines"
    },
    "Plastics & Materials": {
        "1301.TW": "Formosa Plastics",
        "1303.TW": "Nan Ya Plastics",
        "1326.TW": "Formosa Chemicals",
        "6505.TWO": "Formosa Petrochemical",
        "1304.TW": "USI Corporation"
    },
    "Steel & Metals": {
        "2002.TW": "China Steel",
        "2014.TW": "Chung Hung Steel",
        "2006.TW": "Tung Ho Steel",
        "2027.TW": "Ta Chen Stainless",
        "2031.TW": "Sheng Yu Steel"
    },
    "Food & Retail": {
        "1216.TW": "Uni-President",
        "2912.TW": "President Chain Store (7-Eleven)",
        "1210.TW": "Great Wall Enterprise",
        "1215.TW": "Charoen Pokphand Enterprise",
        "1227.TW": "Standard Foods"
    },
    "Cement & Building Materials": {
        "1101.TW": "Taiwan Cement",
        "1102.TW": "Asia Cement",
        "2542.TW": "Highwealth Construction",
        "2501.TW": "Cathay Real Estate",
        "1104.TW": "Universal Cement"
    },
    "Biotech & Healthcare": {
        "1795.TW": "Lotus Pharmaceutical",
        "4743.TWO": "Oneness Biotech",
        "6492.TWO": "Senhwa Biosciences",
        "4147.TWO": "TaiMed Biologics",
        "1752.TW": "Nang Kuang Pharma"
    },
    "Digital & New Economy": {
        "8454.TW": "momo.com (E-commerce)",
        "6180.TWO": "Gamania (Gaming)",
        "8044.TWO": "PChome Online",
        "3045.TW": "Taiwan Mobile",
        "4904.TW": "Far EasTone"
    }
}

# ===============================
# Sidebar｜Session State
# ===============================
if "stocks" not in st.session_state:
    st.session_state.stocks = []

def add_stock():
    st.session_state.stocks.append({
        "sector": None,
        "ticker": None,
        "source": "Yahoo Finance",
        "csv_df": None
    })

for stock in st.session_state.stocks:
    # 如果 source 不存在，給預設值
    if "source" not in stock:
        stock["source"] = "Yahoo Finance"
    if "csv_df" not in stock:
        stock["csv_df"] = None

def remove_stock(index):
    st.session_state.stocks.pop(index)

# 預設至少一支
if len(st.session_state.stocks) == 0:
    add_stock()

# ===============================
# Sidebar｜股票設定
# ===============================
st.sidebar.markdown("### 🏷️ 股票選擇")

for i, stock in enumerate(st.session_state.stocks):
    st.sidebar.markdown(f"#### 股票 {i+1}")

    # -------------------------------
    # Yahoo Finance 模式
    # -------------------------------
    if data_source == "Yahoo Finance":
        sector = st.sidebar.selectbox(
            "產業別",
            list(stock_sector_map.keys()),
            index=list(stock_sector_map.keys()).index(stock["sector"])
            if stock["sector"] in stock_sector_map else 0,
            key=f"sector_{i}"
        )
        stock["sector"] = sector

        # 建立顯示名稱列表：Stock Name (Ticker)
        ticker_list = list(stock_sector_map[sector].keys())
        ticker_display = [f"{stock_sector_map[sector][t]} ({t})" for t in ticker_list]

        # 找目前選中的 index
        if stock["ticker"] in ticker_list:
            selected_index = ticker_list.index(stock["ticker"])
        else:
            selected_index = 0

        # 顯示 selectbox
        selected_display = st.sidebar.selectbox(
            "股票代號 / 名稱",
            ticker_display,
            index=selected_index,
            key=f"ticker_{i}"
        )

        # 將選到的 ticker 存回 session_state
        stock["ticker"] = ticker_list[ticker_display.index(selected_display)]

        stock["csv_df"] = None

    # -------------------------------
    # CSV 上傳模式
    # -------------------------------
    else:
        uploaded_file = st.sidebar.file_uploader(
            "上傳 CSV（需包含 Date, Close）",
            type=["csv"],
            key=f"csv_{i}"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if "Date" not in df.columns or "Close" not in df.columns:
                    st.sidebar.error("❌ CSV 必須包含 Date 與 Close")
                else:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date")
                    stock["csv_df"] = df
                    stock["sector"] = "CSV"
                    stock["ticker"] = f"CSV_{i+1}"
                    st.sidebar.success("✅ CSV 讀取成功")

            except Exception:
                st.sidebar.error("❌ CSV 讀取失敗")

    # 刪除股票
    if st.sidebar.button("❌ 刪除此股票", key=f"del_{i}"):
        remove_stock(i)
        st.rerun()

# 新增股票
st.sidebar.button("＋ 新增股票", on_click=add_stock)

# ===============================
# Sidebar｜分析期間
# ===============================
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏳ 分析期間")

period_option = st.sidebar.radio("", ["短期", "中期", "長期"])
period_map = {"短期": 90, "中期": 180, "長期": 365}
analysis_days = period_map[period_option]

# ===============================
# Sidebar｜開始分析
# ===============================
st.sidebar.markdown("---")
start_analysis = st.sidebar.button("🚀 開始分析")












# ===============================
# Main Bar｜Step 1
# ===============================
st.title("📈 股票時間序列分析")

if not start_analysis:
    st.info("👈 請先在左側完成設定並點擊「開始分析」")
    st.stop()

if len(st.session_state.stocks) == 0:
    st.warning("⚠️ 尚未選擇任何股票")
    st.stop()

# ===============================
# Subtitle｜股票名稱與代號
# ===============================
stock_titles = []

for i, stock in enumerate(st.session_state.stocks):

    if data_source == "Yahoo Finance":
        sector = stock["sector"]
        ticker = stock["ticker"]
        name = stock_sector_map[sector][ticker]
        stock_titles.append(f"{name} ({ticker})")
    else:
        stock_titles.append(f"CSV Stock {i+1}")

st.subheader(" / ".join(stock_titles))

st.header("📈 股票時間序列分析 & 技術指標分析")

short_window = 10
long_window = 30
rsi_window = 14 


for i, stock in enumerate(st.session_state.stocks):
    st.markdown("---")  # 分隔不同股票
    st.subheader(f"{stock_titles[i]} 分析")

    # --------------------------
    # 取得資料
    # --------------------------
    if data_source == "Yahoo Finance":
        ticker = stock["ticker"]
        end_date = pd.Timestamp.today()
        start_date = end_date - timedelta(days=analysis_days)

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.warning("⚠️ 無法取得資料")
            continue
        df = df.reset_index()

    else:  # CSV
        df = stock.get("csv_df")
        if df is None or df.empty:
            st.warning("⚠️ CSV 無資料")
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        end_date = df["Date"].max()
        start_date = end_date - timedelta(days=analysis_days)
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if df.empty:
            st.warning("⚠️ CSV 在分析期間沒有資料")
            continue

    # --------------------------
    # 股票價格繪圖
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="black")
    ax.set_title(f"{stock_titles[i]} - 股價走勢")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --------------------------
    # 計算 SMA
    # --------------------------
    df["SMA_short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_long"] = df["Close"].rolling(window=long_window).mean()

    # --------------------------
    # SMA 圖表
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="black")
    ax.plot(df["Date"], df["SMA_short"], label=f"SMA {short_window}", color="blue")
    ax.plot(df["Date"], df["SMA_long"], label=f"SMA {long_window}", color="red")
    ax.set_title(f"{stock_titles[i]} - SMA 分析")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --------------------------
    # 趨勢分析文字
    # --------------------------
    latest_close = float(df["Close"].iloc[-1])
    latest_sma_short = float(df["SMA_short"].iloc[-1])
    latest_sma_long = float(df["SMA_long"].iloc[-1])

    if not pd.isna(latest_sma_short) and not pd.isna(latest_sma_long):
        if latest_sma_short > latest_sma_long:
            trend_msg = f"📈 短期 SMA 高於長期 SMA，近期趨勢向上。"
        elif latest_sma_short < latest_sma_long:
            trend_msg = f"📉 短期 SMA 低於長期 SMA，近期趨勢向下。"
        else:
            trend_msg = f"⚖️ 短期 SMA 與長期 SMA 接近，盤整中。"

        if latest_close > latest_sma_short:
            trend_msg += f"\n➡️ 最新收盤價 {latest_close:.2f} 高於短期 SMA {latest_sma_short:.2f}，短線動能偏強。"
        else:
            trend_msg += f"\n➡️ 最新收盤價 {latest_close:.2f} 低於短期 SMA {latest_sma_short:.2f}，短線壓力偏大。"

        st.markdown(trend_msg)
    else:
        st.markdown("⚠️ SMA 尚未有足夠資料計算")


    # --------------------------
    # 計算 RSI (確保 1D)
    # --------------------------
    close_series = df["Close"].squeeze()
    df["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=rsi_window).rsi()


    # --------------------------
    # RSI 圖表
    # --------------------------

    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["Date"], df["RSI"], label=f"RSI {rsi_window}", color="purple")
    ax.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    ax.set_title(f"{stock_titles[i]} - RSI 技術指標")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --------------------------
    # 小分析文字
    # --------------------------
    latest_rsi = float(df["RSI"].iloc[-1]) if not df["RSI"].empty else None
    trend_msg = ""
    if latest_rsi is not None:
        trend_msg += f"📊 最新 RSI({rsi_window}) 值: {latest_rsi:.2f}\n"
        if latest_rsi > 70:
            trend_msg += "⚠️ RSI 高於 70，短線可能過買，價格短期可能回檔。\n"
        elif latest_rsi < 30:
            trend_msg += "✅ RSI 低於 30，短線可能過賣，價格短期可能反彈。\n"
        else:
            trend_msg += "ℹ️ RSI 在中間區間，短線動能中性。\n"

        # 結合 SMA + RSI 小建議
        if latest_close > latest_sma_short and latest_rsi < 70:
            trend_msg += "➡️ 短線上升動能存在，可能適合持有或觀察買入。\n"
        elif latest_close < latest_sma_short and latest_rsi > 30:
            trend_msg += "➡️ 短線動能偏弱，建議謹慎操作。\n"

    st.markdown(trend_msg)

    
    # --------------------------
    # 4a.3 計算 MACD
    # --------------------------
    # MACD 參數
    fast_window = 12
    slow_window = 26
    signal_window = 9

    # 計算 EMA
    df["EMA_fast"] = df["Close"].ewm(span=fast_window, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow_window, adjust=False).mean()

    # MACD 線與信號線
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --------------------------
    # MACD 圖表
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["Date"], df["MACD"], label="MACD", color="blue")
    ax.plot(df["Date"], df["MACD_Signal"], label="Signal", color="red")
    ax.bar(df["Date"], df["MACD_Hist"], label="Histogram", color="gray", alpha=0.5)
    ax.set_title(f"{stock_titles[i]} - MACD 技術指標")
    ax.set_xlabel("Date")
    ax.set_ylabel("MACD")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --------------------------
    # MACD 小分析
    # --------------------------
    latest_macd = df["MACD"].iloc[-1]
    latest_signal = df["MACD_Signal"].iloc[-1]

    macd_msg = f"📊 最新 MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f}\n"

    if latest_macd > latest_signal:
        macd_msg += "➡️ MACD 線在 Signal 線上方，短線偏多，動能可能向上。\n"
    elif latest_macd < latest_signal:
        macd_msg += "➡️ MACD 線在 Signal 線下方，短線偏空，動能可能向下。\n"
    else:
        macd_msg += "➡️ MACD 線與 Signal 線接近，短線動能中性。\n"

    # 進一步結合 histogram
    latest_hist = df["MACD_Hist"].iloc[-1]
    if latest_hist > 0:
        macd_msg += "➡️ Histogram 為正值，短期上漲動能可能增強。\n"
    elif latest_hist < 0:
        macd_msg += "➡️ Histogram 為負值，短期下跌壓力可能增加。\n"
    else:
        macd_msg += "➡️ Histogram 接近零，短期動能偏中性。\n"

    st.markdown(macd_msg)

    # --------------------------
    # 4a.4 綜合技術指標分析（Buy / Hold / Sell）
    # --------------------------

    st.subheader("📌 綜合技術指標分析 (SMA + RSI + MACD)")

    # ===== 計算 SMA 分數 =====
    sma_score = 0
    sma_note_list = []
    if latest_sma_short > latest_sma_long:
        sma_score += 1
        sma_note_list.append("短期 SMA > 長期 SMA → +1分")
    elif latest_sma_short < latest_sma_long:
        sma_score -= 1
        sma_note_list.append("短期 SMA < 長期 SMA → -1分")

    if latest_close > latest_sma_short:
        sma_score += 1
        sma_note_list.append("收盤價 > 短期 SMA → +1分")
    elif latest_close < latest_sma_short:
        sma_score -= 1
        sma_note_list.append("收盤價 < 短期 SMA → -1分")

    # ===== 計算 RSI 分數 =====
    rsi_score = 0
    rsi_note_list = []
    if latest_rsi < 30:
        rsi_score += 1
        rsi_note_list.append("RSI < 30 → 過賣 → +1分")
    elif latest_rsi > 70:
        rsi_score -= 1
        rsi_note_list.append("RSI > 70 → 過買 → -1分")
    else:
        rsi_note_list.append("RSI 在 30~70 → 中性 → 0分")

    # ===== 計算 MACD 分數 =====
    macd_score = 0
    macd_note_list = []
    if latest_macd > latest_signal:
        macd_score += 1
        macd_note_list.append("MACD > Signal → +1分")
    elif latest_macd < latest_signal:
        macd_score -= 1
        macd_note_list.append("MACD < Signal → -1分")

    if latest_hist > 0:
        macd_score += 0.5
        macd_note_list.append("Histogram > 0 → +0.5分")
    elif latest_hist < 0:
        macd_score -= 0.5
        macd_note_list.append("Histogram < 0 → -0.5分")
    else:
        macd_note_list.append("Histogram ≈ 0 → 0分")

    # ==========================
    # 各指標計分明細（巢狀 bullet）
    # ==========================
    st.markdown("**各指標計分明細:**")

    sma_lines = "\n".join([f"    - {line}" for line in sma_note_list])
    st.markdown(f"- **SMA 分數:** {sma_score}\n{sma_lines}")

    rsi_lines = "\n".join([f"    - {line}" for line in rsi_note_list])
    st.markdown(f"- **RSI 分數:** {rsi_score}\n{rsi_lines}")

    macd_lines = "\n".join([f"    - {line}" for line in macd_note_list])
    st.markdown(f"- **MACD 分數:** {macd_score}\n{macd_lines}")

    # ==========================
    # 總分與操作建議
    # ==========================
    total_score = sma_score + rsi_score + macd_score

    if total_score >= 2:
        recommendation = "🟢 Buy"
    elif total_score <= -2:
        recommendation = "🔴 Sell"
    else:
        recommendation = "🟡 Hold"

    st.markdown(f"**總分: {total_score} → 建議操作: {recommendation}**")

    # ==========================
    # 隱藏邏輯說明 (expander)
    # ==========================
    with st.expander("📖 查看詳細計分邏輯"):
        logic_text = """
    **SMA (短期 vs 長期)**

    - 短期 SMA > 長期 SMA → 多頭 → +1 分
    - 短期 SMA < 長期 SMA → 空頭 → -1 分
    - 收盤價 > 短期 SMA → 動能存在 → +1 分
    - 收盤價 < 短期 SMA → 動能偏弱 → -1 分

    **RSI**

    - RSI < 30 → 過賣 → 多頭 → +1 分
    - RSI > 70 → 過買 → 空頭 → -1 分
    - RSI 在 30~70 → 中性 → 0 分

    **MACD**

    - MACD > Signal → 多頭 → +1 分
    - MACD < Signal → 空頭 → -1 分
    - Histogram 正值 → 強化多頭 → +0.5 分
    - Histogram 負值 → 強化空頭 → -0.5 分
    """
        st.markdown(logic_text)

    
    # ==========================
    # 4b. SARIMAX 1 個月未來預測 (自動選參數)
    # ==========================

    st.subheader("📊 SARIMAX 1 個月股價預測（自動選最佳參數）")

    # 用歷史收盤價建模
    ts = df.set_index("Date")["Close"]

    # 處理缺失值
    ts = ts.interpolate()  # 線性補缺失值
    ts = ts.asfreq('B')    # 轉為交易日頻率，非交易日自動補 NaN 後用 interpolate 填充
    ts = ts.fillna(method='ffill')

    try:
        # -------------------------
        # 1️⃣ 自動選擇最佳 ARIMA (p,d,q)
        # -------------------------
        auto_model = pm.auto_arima(ts,
                                seasonal=False,
                                stepwise=True,
                                suppress_warnings=True,
                                error_action='ignore')
        best_order = auto_model.order  # (p,d,q)
        st.markdown(f"➡️ 自動選擇最佳參數: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}")

        # -------------------------
        # 2️⃣ 建立 SARIMAX 模型 (Fit)
        # -------------------------
        sarimax_model = SARIMAX(ts,
                                order=best_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        sarimax_fit = sarimax_model.fit(disp=False)

        # -------------------------
        # 3️⃣ 預測未來 30 個交易日 (Forecast)
        # -------------------------
        forecast_steps = 30
        forecast_res = sarimax_fit.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        # 建立未來日期 index (交易日)
        last_date = ts.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                    periods=forecast_steps, freq='B')
        forecast_series = pd.Series(forecast_mean.values, index=forecast_dates)
        conf_lower = pd.Series(conf_int.iloc[:,0].values, index=forecast_dates)
        conf_upper = pd.Series(conf_int.iloc[:,1].values, index=forecast_dates)

        # -------------------------
        # 4️⃣ 繪圖：歷史 + 預測
        # -------------------------
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(ts.index, ts.values, label="歷史收盤價", color="black")
        ax.plot(forecast_series.index, forecast_series.values, label="SARIMAX 預測", color="orange")
        ax.fill_between(forecast_series.index, conf_lower, conf_upper, color='orange', alpha=0.2, label="95% 信賴區間")
        ax.set_title(f"{stock_titles[i]} - SARIMAX 未來 1 個月預測")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # -------------------------
        # 5️⃣ 顯示文字分析
        # -------------------------
        st.markdown(f"➡️ SARIMAX 預測價格範圍（近 1 個月）：從 {forecast_series.min():.2f} 到 {forecast_series.max():.2f}")

    except Exception as e:
        st.warning(f"⚠️ SARIMAX 預測失敗: {e}")

    
    # ===========================
    # 4C. 中文新聞 (NewsAPI) + 影響分析
    # ===========================

    st.subheader("📰 中文新聞 (NewsAPI) & 影響分析")

    # 取得中文公司名稱
    if data_source == "Yahoo Finance":
        company_name_eng = stock_sector_map[stock["sector"]][stock["ticker"]]
        company_name_cn = stock_eng_to_cn.stock_eng_to_cn.get(company_name_eng, company_name_eng)
    else:
        # CSV 模式直接用 ticker 名稱
        company_name_cn = stock["ticker"]

    st.markdown(f"📌 搜尋公司: **{company_name_cn}**")

    # NewsAPI 設計查詢
    news_api_key = "484de4e34b174e3ebf41ddf14bdeb201"  # 請替換成你自己的 key
    query = company_name_cn  # 用中文名稱搜尋

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=zh&sortBy=publishedAt&pageSize=3&apiKey={news_api_key}"
    )

    try:
        res = requests.get(url).json()
        articles = res.get("articles", [])

        if not articles:
            st.info("⚠️ 找不到相關中文新聞")
        else:
            # 顯示新聞
            for idx, art in enumerate(articles, start=1):
                title = art.get("title", "")
                desc = art.get("description", "")
                link = art.get("url", "")
                published = art.get("publishedAt", "")

                st.markdown(f"**{idx}. [{title}]({link})**")
                st.markdown(f"*{published}*")
                st.markdown(desc or "（無摘要）")

            # 簡易情緒影響判讀
            st.markdown("📊 新聞可能對股價影響判斷:")

            for idx, art in enumerate(articles, start=1):
                title = art.get("title", "")
                desc = art.get("description", "")
                combined = f"{title} {desc}".lower()

                # 簡單正負面關鍵字
                positive = ["利多", "上漲", "創高", "成長", "恢復", "擴大"]
                negative = ["利空", "下跌", "虧損", "調降", "衰退", "壓力"]

                pos_score = sum([combined.count(w) for w in positive])
                neg_score = sum([combined.count(w) for w in negative])

                if pos_score > neg_score:
                    impact = "🟢 正面 → 可能推高股價"
                elif neg_score > pos_score:
                    impact = "🔴 負面 → 可能壓低股價"
                else:
                    impact = "⚪ 中性 → 影響不明顯"

                st.markdown(f"- 新聞 {idx}: {impact} (正面語詞: {pos_score}, 負面語詞: {neg_score})")

    except Exception as e:
        st.error(f"⚠️ 取得或解析新聞時出錯: {e}")

        

# ==========================
# 4d. RSI 回測分析（金融績效指標）
# ==========================
st.subheader("📊 RSI 回測與金融績效分析 (過去 1 年)")

rsi_backtest_window = 14
backtest_days = 365

performance_data = {}

for i, stock in enumerate(st.session_state.stocks):
    st.markdown("---")
    st.subheader(f"{stock_titles[i]} RSI 回測分析")

    # 取得過去一年資料
    if data_source == "Yahoo Finance":
        ticker = stock["ticker"]
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=backtest_days)
        df_bt = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df_bt.empty:
            st.warning("⚠️ 無法取得資料")
            continue
        df_bt = df_bt.reset_index()
    else:  # CSV 模式
        df_bt = stock.get("csv_df")
        if df_bt is None or df_bt.empty:
            st.warning("⚠️ CSV 無資料")
            continue

        df_bt["Date"] = pd.to_datetime(df_bt["Date"])
        df_bt = df_bt.sort_values("Date")

        # 用 CSV 最新日期作為「今天」
        latest_csv_date = df_bt["Date"].max()
        start_date = latest_csv_date - pd.Timedelta(days=backtest_days)

        # 過濾回測期間資料
        df_bt = df_bt[df_bt["Date"] >= start_date]
        if df_bt.empty:
            st.warning("⚠️ CSV 在回測期間無資料")
            continue


    # 計算 RSI
    df_bt["RSI"] = ta.momentum.RSIIndicator(close=df_bt["Close"].squeeze(), window=rsi_backtest_window).rsi()

    # --------------------------
    # 根據 RSI 訊號生成持倉策略
    # --------------------------
    df_bt["Position"] = 0
    df_bt.loc[df_bt["RSI"] < 30, "Position"] = 1   # RSI 過賣 → 買入
    df_bt.loc[df_bt["RSI"] > 70, "Position"] = -1  # RSI 過買 → 賣出
    df_bt["Position"] = df_bt["Position"].ffill().shift(1).fillna(0)  # 前一天持倉，避免當天交易未反映

    # --------------------------
    # 計算每日收益率與策略累積收益率
    # --------------------------
    df_bt["Daily_Return"] = df_bt["Close"].pct_change()
    df_bt["Strategy_Return"] = df_bt["Daily_Return"] * df_bt["Position"]
    df_bt["Cumulative_Return"] = (1 + df_bt["Strategy_Return"]).cumprod() - 1
    df_bt["Buy_Hold_Cumulative"] = (1 + df_bt["Daily_Return"]).cumprod() - 1

    # --------------------------
    # 總結績效指標
    # --------------------------
    daily_return_mean = df_bt["Strategy_Return"].mean()
    cumulative_return = df_bt["Cumulative_Return"].iloc[-1]
    volatility = df_bt["Strategy_Return"].std() * np.sqrt(252)

    performance_data[stock_titles[i]] = {
        "Daily_Return": daily_return_mean,
        "Cumulative_Return": cumulative_return,
        "Volatility": volatility,
        "df": df_bt
    }

    st.markdown(f"- 平均每日收益率: {daily_return_mean:.4%}")
    st.markdown(f"- 累積收益率: {cumulative_return:.2%}")
    st.markdown(f"- 波動率 (年化): {volatility:.2%}")

    # --------------------------
    # 繪製累積收益率曲線
    # --------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_bt["Date"], df_bt["Cumulative_Return"], label="RSI 策略累積收益", color="blue")
    ax.plot(df_bt["Date"], df_bt["Buy_Hold_Cumulative"], label="Buy & Hold 累積收益", color="orange", linestyle="--")
    ax.set_title(f"{stock_titles[i]} - RSI 策略 vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("累積收益率")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --------------------------
# 多支股票比較
# --------------------------
if len(performance_data) > 1:
    st.markdown("---")
    st.subheader("📊 多支股票 RSI 策略績效比較")

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, data in performance_data.items():
        ax.plot(data["df"]["Date"], data["df"]["Cumulative_Return"], label=f"{name} RSI 策略")
    ax.set_title("RSI 策略累積收益率比較")
    ax.set_xlabel("Date")
    ax.set_ylabel("累積收益率")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 比較總累積收益率與波動率
    summary_text = "📌 總結分析:\n"
    sorted_by_return = sorted(performance_data.items(), key=lambda x: x[1]["Cumulative_Return"], reverse=True)
    sorted_by_vol = sorted(performance_data.items(), key=lambda x: x[1]["Volatility"])
    summary_text += f"- 累積收益率最高: {sorted_by_return[0][0]} ({sorted_by_return[0][1]['Cumulative_Return']:.2%})\n"
    summary_text += f"- 波動率最低 (最穩定): {sorted_by_vol[0][0]} ({sorted_by_vol[0][1]['Volatility']:.2%})\n"

    st.markdown(summary_text)
