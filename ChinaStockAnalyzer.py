# investment_analyzer_china.py
import akshare as ak
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from datetime import timedelta
from functools import reduce
import time
import ta
# from streamlit_js_eval import streamlit_js_eval

import urllib3
urllib3.disable_warnings()

import akshare as ak
import requests
requests.packages.urllib3.disable_warnings()  # 关闭警告


# patch akshare 用的 requests，使其不验证证书
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = patched_get
requests.Session.get = patched_get


st.set_page_config(page_title="中国股票分析器", layout="wide")
st.title("📊 中国股票分析器")

# --- Sidebar Input ---
st.sidebar.header("投资组合输入")
uploaded_file = st.sidebar.file_uploader("上传你的投资组合CSV文件", type=["csv"])
st.sidebar.markdown("示例格式(**股票代码无需市场前缀**):")
st.sidebar.code("Ticker,Name\n600519,贵州茅台\n000001,平安银行")

# --- Default Example Portfolio ---
default_portfolio = pd.DataFrame({
    "Ticker": ["600519", "000001"],
    "Name": ["贵州茅台", "平安银行"]
})

try:
    # 分别抓取主板a股、科创板、深市a股
    df_za = ak.stock_info_sh_name_code("主板A股")  # 上交所主板
    df_kc = ak.stock_info_sh_name_code("科创板")   # 上交所科创板
    df_a = ak.stock_info_sz_name_code("A股列表")   # 深交所A股

    # 统一列名，防止列名不一致
    for df in [df_za, df_kc, df_a]:
        df.rename(columns=lambda x: x.strip(), inplace=True)  # 去除列名空格
        df.rename(columns={"证券代码": "代码", "证券简称": "名称"}, inplace=True)

    # 合并
    df_all = reduce(lambda a, b: pd.concat([a, b], ignore_index=True), [df_za, df_kc, df_a])

    # 填充代码格式 & 映射
    df_all["code"] = df_all["代码"].astype(str).str.zfill(6)
    code_name_map = dict(zip(df_all["code"], df_all["名称"]))

except Exception as e:
    st.warning(f"无法获取股票名称映射: {e}")
    code_name_map = {}

# --- Read uploaded file or use default ---
if uploaded_file:
    try:
        portfolio = pd.read_csv(uploaded_file, dtype=object)
        st.success("投资组合上传成功")
    except Exception as e:
        st.error(f"读取CSV错误: {e}")
        st.stop()
else:
    st.info("使用示例投资组合")
    portfolio = default_portfolio.copy()

# --- Add Name column if missing ---
if "Name" not in portfolio.columns:
    portfolio["Name"] = ""

# --- Build stock name ---
def get_stock_name(row):
    code = str(row["Ticker"]).zfill(6)
    # Prefer AkShare mapping, fallback to uploaded Name, finally "未知"
    return code_name_map.get(code) or row.get("Name") or "未知"

portfolio["Name"] = portfolio.apply(get_stock_name, axis=1)

# --- Fetch Data ---
st.subheader("投资组合概览")
results = []
price_data = {}
buy_sell_suggestions = []
special_signals = []

RETRIES = 3
PAUSE_SECONDS = 2

for ticker in portfolio["Ticker"]:
    try:
        # Retry logic for stock_zh_a_hist
        df = None
        for attempt in range(RETRIES):
            try:
                df = ak.stock_zh_a_hist(symbol=ticker, adjust="qfq")
                break
            except Exception as inner_e:
                if attempt < RETRIES - 1:
                    time.sleep(PAUSE_SECONDS)
                else:
                    raise inner_e

        df["date"] = pd.to_datetime(df["日期"])
        df.set_index("date", inplace=True)
        df = df.sort_index()

        # Moving averages for 2560策略
        short_ma = df["收盘"].rolling(window=5).mean()
        middle_ma = df["收盘"].rolling(window=25).mean()
        long_ma = df["收盘"].rolling(window=60).mean()

        # 计算均线
        ma5_now = short_ma.iloc[-1]
        ma5_prev = short_ma.iloc[-2]
        ma25_now = middle_ma.iloc[-1]
        ma25_prev = middle_ma.iloc[-2]
        ma60_now = long_ma.iloc[-1]
        ma60_prev = long_ma.iloc[-2]
        
        vol_ma5_now = df["成交量"].rolling(window=5).mean().iloc[-1]
        vol_ma60_now = df["成交量"].rolling(window=60).mean().iloc[-1]
        
        # 25日均线趋势
        trend_25 = "上升" if ma25_now > ma25_prev else "下降"
        
        signal_2560 = "2560策略: 数据不足"
        
        # 2560策略买入条件
        if (
            not np.isnan(ma5_now) and not np.isnan(ma5_prev) and not np.isnan(ma25_now) and not np.isnan(ma25_prev)
            and not np.isnan(vol_ma5_now) and not np.isnan(vol_ma60_now)
        ):
            if (ma25_now > ma25_prev):  # 25日均线上升
                # 5日均线上穿25日均线，且量能满足
                if (ma5_prev < ma25_prev) and (ma5_now > ma25_now) and (vol_ma5_now > vol_ma60_now):
                    signal_2560 = "📈 2560策略: 满足所有条件，短线买入信号（25日均线上升, 5日均线上穿25日, 5日量能>60日量能）"
                # 5日均线回踩25日均线，且量能满足
                elif (abs(ma5_now - ma25_now) < 1e-4) and (vol_ma5_now > vol_ma60_now):
                    signal_2560 = "📈 2560策略: 满足回踩条件，关注买入（25日均线上升, 5日回踩25日, 5日量能>60日量能）"
                else:
                    signal_2560 = f"2560策略: 当前25日均线趋势{trend_25}，条件未全部满足"
            else:
                signal_2560 = f"2560策略: 当前25日均线趋势{trend_25}，条件未全部满足"
        else:
            signal_2560 = "2560策略: 数据不足"

        # Technical indicators
        daily_returns = df["收盘"].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        price_data[ticker] = df["收盘"]
        current_price = df["收盘"].iloc[-1]

        delta = df["收盘"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]

        exp1 = df["收盘"].ewm(span=12, adjust=False).mean()
        exp2 = df["收盘"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = macd.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_prev = signal.iloc[-2]

        obv = df["成交量"].copy()
        obv[df["收盘"].diff() < 0] *= -1
        obv = obv.cumsum()
        obv_trend = obv.diff().mean()
        vol_spike = df["成交量"].iloc[-1] > 1.5 * df["成交量"].rolling(10).mean().iloc[-1]

        signal_match = (
            macd_prev < signal_prev and
            macd_value > signal.iloc[-1] and
            rsi_value < 70 and
            obv_trend > 0 and
            vol_spike
        )

        # ----- Operation Suggestions -----
        suggestion = "🔍 继续观察走势"
        # --- Target Calculation (Merged Robust Logic) ---
        N = 5  # short-term window
        min_pct = 0.02
        max_pct = 0.10
        
        # Daily returns and volatility
        daily_returns = df["收盘"].pct_change().dropna()
        current_price = df["收盘"].iloc[-1]
        
        if len(daily_returns) < N:
            vol_target_pct = 0.05
            nan_flag = True
        else:
            daily_vol = daily_returns.std()
            vol_target_pct = daily_vol * np.sqrt(N)
            vol_target_pct = np.clip(vol_target_pct, min_pct, max_pct)
            nan_flag = np.isnan(vol_target_pct)
        
        # ATR as alternative (with try/except)
        try:
            atr_series = ta.volatility.AverageTrueRange(
                high=df['最高'],
                low=df['最低'],
                close=df['收盘'],
                window=14
            ).average_true_range()
            atr = atr_series.iloc[-1]
            if np.isnan(atr):
                atr = None
            atr_target_price_up = current_price + (atr * N) if atr is not None else None
            atr_target_price_down = current_price - (atr * N) if atr is not None else None
        except Exception as e:
            atr = None
            atr_target_price_up = None
            atr_target_price_down = None

        # --- OBV Calculation ---
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=df["收盘"],
            volume=df["成交量"]
        ).on_balance_volume()

        df["obv_ma34"] = df["obv"].rolling(window=34).mean()

        # --- Generate OBV 34 Buy/Sell Signals ---
        df["obv_signal"] = 0
        df["obv_signal"] = np.where(df["obv"] > df["obv_ma34"], 1, 0)  # 1 = above (potential buy)
        df["obv_signal"] = np.where(df["obv"] < df["obv_ma34"], -1, df["obv_signal"])  # -1 = below (potential sell)
        # Optionally, find signal changes (crossovers)
        df["obv_trade_signal"] = df["obv_signal"].diff()
        latest_signal = df["obv_trade_signal"].dropna().iloc[-1]
        
        # Nearest resistance/support (20-day high/low)
        resistance = df['收盘'].rolling(window=20).max().iloc[-2] if len(df) >= 21 else None
        support = df['收盘'].rolling(window=20).min().iloc[-2] if len(df) >= 21 else None
        
        # --- Target Suggestion Logic (with robust fallback) ---
        if nan_flag or np.isnan(vol_target_pct):
            target_str = "数据不足, 使用默认5%"
            target_pct_display = 5.0
        else:
            target_str = f"按历史波动率{vol_target_pct*100:.1f}%"
            target_pct_display = vol_target_pct * 100
        # --- Buy signal ---
        if (short_ma.iloc[-1] > long_ma.iloc[-1] and rsi_value < 70 and macd_value > signal.iloc[-1]) or (latest_signal == 2):
            vol_based_up = current_price * (1 + vol_target_pct) if not np.isnan(current_price) and not np.isnan(vol_target_pct) else None
            targets_up = [vol_based_up]
            if atr_target_price_up:
                targets_up.append(atr_target_price_up)
            if resistance and resistance > current_price:
                targets_up.append(resistance)
            # Filter out None/nan
            targets_up_valid = [x for x in targets_up if x is not None and not np.isnan(x)]
            if targets_up_valid:
                final_target_up = min(targets_up_valid)
            else:
                final_target_up = current_price * 1.05  # fallback
            suggestion = (
                f"📈 建议关注买入机会 (动态目标价约 ¥{final_target_up:.2f}, {target_str})"
                + (f"\nATR估算目标: ¥{atr_target_price_up:.2f}" if atr_target_price_up else "")
                + (f"\n技术阻力位: ¥{resistance:.2f}" if resistance else "")
            )
        # --- Sell signal ---
        elif (short_ma.iloc[-1] < long_ma.iloc[-1] and rsi_value > 70 and macd_value < signal.iloc[-1]) or (latest_signal == -2):
            vol_based_down = current_price * (1 - vol_target_pct) if not np.isnan(current_price) and not np.isnan(vol_target_pct) else None
            targets_down = [vol_based_down]
            if atr_target_price_down:
                targets_down.append(atr_target_price_down)
            if support and support < current_price:
                targets_down.append(support)
            targets_down_valid = [x for x in targets_down if x is not None and not np.isnan(x)]
            if targets_down_valid:
                final_target_down = max(targets_down_valid)
            else:
                final_target_down = current_price * 0.95  # fallback
            suggestion = (
                f"📉 建议考虑止盈或卖出 (动态支撑位约 ¥{final_target_down:.2f}, {target_str})"
                + (f"\nATR估算支撑: ¥{atr_target_price_down:.2f}" if atr_target_price_down else "")
                + (f"\n技术支撑位: ¥{support:.2f}" if support else "")
            )
        # --- Neutral/hold ---
        else:
            suggestion = "🔍 继续观察走势"

        # Combine with 2560策略
        suggestion = f"{suggestion}\n{signal_2560}"

        if latest_signal == 2:
            obv_signal = "📈 OBV金叉（突破34日均线）"
        elif latest_signal == -2:
            obv_signal = "📉 OBV死叉（跌破34日均线）"
        else:
            obv_signal = "🔍 OBV维持趋势中"

        buy_sell_suggestions.append({
            "代码": ticker,
            "名称": portfolio.loc[portfolio["Ticker"] == ticker, "Name"].values[0],
            "操作建议": suggestion,
            "RSI": round(rsi_value, 2),
            "MACD": round(macd_value, 2),
            "MA5": round(ma5_now, 2) if not pd.isna(ma5_now) else "",
            "MA25": round(ma25_now, 2) if not pd.isna(ma25_now) else "",
            "MA60": round(ma60_now, 2) if not pd.isna(ma60_now) else "",
            "OBV 34": obv_signal
        })

        # ----- Special Signals -----
        if signal_match:
            special_signals.append({
                "代码": ticker,
                "名称": portfolio.loc[portfolio["Ticker"] == ticker, "Name"].values[0],
                "当前价格": current_price,
                "RSI": round(rsi_value, 2),
                "MACD": round(macd_value, 2),
                "OBV变化": round(obv_trend, 2),
                "成交量突增": "是" if vol_spike else "否",
                "MA5": round(ma5_now, 2) if not pd.isna(ma5_now) else "",
                "MA25": round(ma25_now, 2) if not pd.isna(ma25_now) else "",
                "MA60": round(ma60_now, 2) if not pd.isna(ma60_now) else ""
            })

        # 只用最近1年（252个交易日）的数据
        if len(df) > 252:
            df_last_year = df.iloc[-252:]
        else:
            df_last_year = df.copy()
        
        start_price = df_last_year["收盘"].iloc[0]
        end_price = df_last_year["收盘"].iloc[-1]
        
        一年回报率 = (end_price / start_price - 1) * 100
        daily_returns_last_year = df_last_year["收盘"].pct_change().dropna()
        年化波动率 = daily_returns_last_year.std() * np.sqrt(252) * 100

        # ----- Overview Table -----
        results.append({
            "代码": ticker,
            "名称": portfolio.loc[portfolio["Ticker"] == ticker, "Name"].values[0],
            "当前价格": current_price,
            "一年回报率 %": 一年回报率,
            "年化波动率 %": 年化波动率,
            "夏普比率": sharpe_ratio
        })

    except Exception as e:
        st.warning(f"获取 {ticker} 数据时出错: {e}")

# --- Display Table ---
result_df = pd.DataFrame(results)
st.dataframe(result_df)

# --- 操作建议 ---
st.subheader("📌 操作建议")
suggestion_df = pd.DataFrame(buy_sell_suggestions)
st.dataframe(suggestion_df)

# --- 特殊技术形态筛选 ---
if special_signals:
    st.subheader("🚀 技术形态筛选结果 (RSI+MACD金口+OBV上升+收量放量)")
    special_df = pd.DataFrame(special_signals)
    st.dataframe(special_df)
else:
    special_df = pd.DataFrame()

# --- AI Analysis ---
st.subheader("🧠 AI 投资建议")
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"] if "DEEPSEEK_API_KEY" in st.secrets else None
if deepseek_api_key:
    client = OpenAI(
        api_key=openai_api_key
    )
    MODEL = "gpt-4o-mini"
    if st.button("🗞 使用 AI 分析投资组合"):
        with st.spinner("正在生成 AI 分析，请稍候..."):
            prompt = f"""
请用中文总结以下中国股票投资组合的投资表现，股票代码无需市场前缀, 并结合市场情绪、短期技术指标（均线、RSI、MACD）, OBV指标、成交量变化等提出操作建议：

[投资概览]
{result_df.to_string(index=False)}

[操作建议]
{suggestion_df.drop(columns=["操作建议"]).to_string(index=False)}

[技术筛选匹配]
{special_df.to_string(index=False)}

如果你知道近期A股市场的热门题材，可以推荐可能相关的个股股票代码等建议。
"""
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "你是一位有经验的中文金融顾问，擅长结合市场热点进行股票筛选。股票代码无需市场前缀."},
                        {"role": "user", "content": prompt.strip()}
                    ],
                    temperature=0.3,
                    max_tokens=1800,
                    stream=False
                )
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"API error: {e}")
else:
    st.info("未配置 API 密钥。请在 Streamlit secrets 中添加以启用 AI 分析。")

# # --- Download Combined Report ---
# st.subheader("📅 下载分析结果")
# st.download_button("下载CSV报告", result_df.to_csv(index=False), file_name="china_portfolio_report.csv")
