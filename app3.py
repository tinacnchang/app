
import streamlit as st
import pandas as pd
import ta
import platform
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import yfinance as yf
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# ------------------------
# 1. 基礎設定與字體修復
# ------------------------
st.set_page_config(page_title="股票時間序列分析", layout="wide")

if platform.system() == "Windows":
    CHINESE_FONT = 'Microsoft JhengHei'
elif platform.system() == "Darwin":
    CHINESE_FONT = 'Heiti TC'
else:
    CHINESE_FONT = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = [CHINESE_FONT]
plt.rcParams['axes.unicode_minus'] = False

# ------------------------
# 2. Session State 狀態管理 (關鍵修改)
# ------------------------
# 用於儲存動態新增的股票清單
if 'stock_rows' not in st.session_state:
    st.session_state.stock_rows = [0]  # 預設有一組輸入框

def add_row():
    st.session_state.stock_rows.append(len(st.session_state.stock_rows))

def remove_row():
    if len(st.session_state.stock_rows) > 1:
        st.session_state.stock_rows.pop()

# ------------------------
# 3. Sidebar 側邊欄設計
# ------------------------
st.sidebar.title("📊 期末專案 - 時間序列分析")
st.sidebar.markdown("**G140A006 葉天明**")
st.sidebar.markdown("---")

stock_sector_map = {
    "Semiconductors (Tech)": {"2330.TW": "台積電", "2454.TW": "聯發科"},
    "AI & Computer Hardware": {"2317.TW": "鴻海", "2382.TW": "廣達"},
    "Financials & Banking": {"2881.TW": "富邦金", "2882.TW": "國泰金"}
}

selected_tickers = []

st.sidebar.markdown("### 🏷️ 股票選擇")
for i in st.session_state.stock_rows:
    with st.sidebar.expander(f"股票設定 {i+1}", expanded=True):
        sector = st.selectbox(f"產業別", list(stock_sector_map.keys()), key=f"sec_{i}")
        ticker = st.selectbox(f"代號", list(stock_sector_map[sector].keys()), key=f"tick_{i}")
        selected_tickers.append(ticker)

col1, col2 = st.sidebar.columns(2)
col1.button("➕ 新增", on_click=add_row)
col2.button("➖ 刪除", on_click=remove_row)

# ------------------------
# 4. 主畫面分析邏輯
# ------------------------
st.title("📈 股票技術分析與策略回測")

if st.sidebar.button("🚀 開始跑分析", type="primary"):
    if not selected_tickers:
        st.warning("請先選擇股票")
    else:
        performance_data = {}
        
        # 建立分欄顯示每支股票的詳細資訊
        for ticker in selected_tickers:
            st.write(f"### 🔍 分析標的：{ticker}")
            
            # 抓取資料 (預設抓取近兩年)
            df = yf.download(ticker, start=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'))
            
            if df.empty:
                st.error(f"無法獲取 {ticker} 的資料")
                continue

            # --- 技術指標計算 ---
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # --- RSI 策略回測 ---
            df['Signal'] = 0
            df.loc[df['RSI'] < 30, 'Signal'] = 1   # 超跌買進
            df.loc[df['RSI'] > 70, 'Signal'] = -1  # 超買賣出
            
            df['Returns'] = df['Close'].pct_change()
            df['Strategy_Return'] = df['Signal'].shift(1) * df['Returns']
            df['Cumulative_Return'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
            df['Buy_Hold_Cumulative'] = (1 + df['Returns'].fillna(0)).cumprod()

            # --- 繪圖 ---
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.index, df["Cumulative_Return"], label="RSI 策略", color="blue")
            ax.plot(df.index, df["Buy_Hold_Cumulative"], label="標的持股 (Buy & Hold)", color="orange", linestyle="--")
            ax.set_title(f"{ticker} 績效回測")
            ax.legend()
            st.pyplot(fig)
            
            performance_data[ticker] = df

        # --- 多股比較圖 ---
        if len(performance_data) > 1:
            st.divider()
            st.subheader("📊 多支股票策略績效總覽")
            fig_all, ax_all = plt.subplots(figsize=(12, 5))
            for name, data in performance_data.items():
                ax_all.plot(data.index, data["Cumulative_Return"], label=name)
            ax_all.set_title("各標的 RSI 策略累積收益率對比")
            ax_all.legend()
            st.pyplot(fig_all)
