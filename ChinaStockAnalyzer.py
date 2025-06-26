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
st.sidebar.code("Ticker\n600519\n000001")

# --- Default Example Portfolio ---
default_portfolio = pd.DataFrame({
    "Ticker": ["600519", "000001"]
})

try:
    # 分别抓取主板A股、科创板、深市A股
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
        portfolio = pd.read_csv(uploaded_file)
        st.success("投资组合上传成功")
    except Exception as e:
        st.error(f"读取CSV错误: {e}")
        st.stop()
else:
    st.info("使用示例投资组合")
    portfolio = default_portfolio.copy()

# --- Calculate Weights from Values ---
# if "Value" not in portfolio.columns or portfolio["Value"].sum() == 0:
#     st.error("投资组合必须包含非零总金额的 'Value' 列")
#     st.stop()
# portfolio["Weight"] = portfolio["Value"] / portfolio["Value"].sum()
portfolio["Name"] = portfolio["Ticker"].astype(str).map(code_name_map)




# --- Fetch Data ---
st.subheader("投资组合概览")
results = []
price_data = {}
buy_sell_suggestions = []


for ticker in portfolio["Ticker"]:
    try:
        df = ak.stock_zh_a_hist(symbol=ticker, adjust="qfq")
        df["date"] = pd.to_datetime(df["日期"])
        df.set_index("date", inplace=True)
        df = df.sort_index()

        daily_returns = df["收盘"].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))

        price_data[ticker] = df["收盘"]

        # 技术指标: 均线策略 + RSI + MACD
        short_ma = df["收盘"].rolling(window=5).mean()
        long_ma = df["收盘"].rolling(window=20).mean()
        latest_price = df["收盘"].iloc[-1]
        short_ma = df["收盘"].rolling(window=5).mean()
        long_ma = df["收盘"].rolling(window=20).mean()
        rsi = 100 - 100 / (1 + df["收盘"].pct_change().rolling(14).mean() / df["收盘"].pct_change().rolling(14).std())

        exp1 = df["收盘"].ewm(span=12, adjust=False).mean()
        exp2 = df["收盘"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # 技术指标: 均线策略 + RSI + MACD
        suggestion = "🔍 继续观察走势"
        target_price = df["收盘"].iloc[-1]

        if len(df) >= 30:
            short_ma = df["收盘"].rolling(window=5).mean()
            long_ma = df["收盘"].rolling(window=20).mean()
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

            if short_ma.iloc[-1] > long_ma.iloc[-1] and rsi.iloc[-1] < 70 and macd.iloc[-1] > signal.iloc[-1]:
                target_price = df["收盘"].iloc[-1] * 1.05
                suggestion = f"📈 建议关注买入机会 (短期目标价约 ¥{target_price:.2f})"
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and rsi.iloc[-1] > 70 and macd.iloc[-1] < signal.iloc[-1]:
                target_price = df["收盘"].iloc[-1] * 0.95
                suggestion = f"📉 建议考虑止盈或卖出 (短期支撑位约 ¥{target_price:.2f})"

        buy_sell_suggestions.append({
            "代码": ticker,
            "名称": portfolio.loc[portfolio["Ticker"] == ticker, "Name"].values[0],
            "操作建议": suggestion,
            "RSI": round(rsi_value, 2) if rsi_value is not None else None,
            "MACD": round(macd_value, 2) if macd_value is not None else None
        })

        results.append({
            "代码": ticker,
            "名称": portfolio.loc[portfolio["Ticker"] == ticker, "Name"].values[0],
            "一年回报率 %": ((df["收盘"].iloc[-1] / df["收盘"].iloc[0]) - 1) * 100,
            "年化波动率 %": volatility * 100,
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


# --- AI Analysis ---
st.subheader("🧠 AI 投资建议")
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
if openai_api_key:
    client = OpenAI(
        api_key=openai_api_key
    )
    
    # Default model; you can change to gpt-4 if you have access
    # MODEL = "gpt-4o-mini"
    MODEL = "gpt-4.1-mini"
    if st.button("🧾 使用 ChatGPT 分析投资组合"):
        prompt = f"请用中文总结以下中国股票投资组合的投资表现，股票代码无需市场前缀, 并结合市场情绪、短期技术指标（均线、RSI、MACD）, 市盈率, 市净率等提出操作建议：\n\n{result_df.to_string(index=False)}\n\n以下是根据技术指标提供的初步建议：\n{suggestion_df.to_string(index=False)}"
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一位有经验的中文金融顾问，善于结合短期市场走势提出操作建议。股票代码无需市场前缀."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1500
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
else:
    st.info("未配置 OpenAI API 密钥。请在 Streamlit secrets 中添加以启用 AI 分析。")

# --- Historical Performance ---
# st.subheader("📈 股票价格表现 (1年归一化)")
# if price_data:
#     df_prices = pd.DataFrame(price_data)
#     df_norm = df_prices / df_prices.iloc[0]
#     df_weighted = df_norm.multiply(portfolio.set_index("Ticker")["Weight"], axis=1)
#     df_portfolio = df_weighted.sum(axis=1)

#     fig, ax = plt.subplots()
#     for col in df_norm.columns:
#         ax.plot(df_norm.index, df_norm[col], alpha=0.4, label=col)
#     ax.plot(df_portfolio.index, df_portfolio, color="black", linewidth=2, label="投资组合")
#     ax.set_title("归一化价格表现 (1年)")
#     ax.set_ylabel("增长倍数")
#     ax.legend()
#     st.pyplot(fig)

# --- Download Combined Report ---
st.subheader("📥 下载分析结果")
st.download_button("下载CSV报告", result_df.to_csv(index=False), file_name="china_portfolio_report.csv")
