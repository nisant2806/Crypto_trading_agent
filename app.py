import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Portfolio state for paper trading
_balance = 10000.0
_portfolio = {}


def _price(coin_id: str) -> float:
    """Fetch the current USD price for a CoinGecko id, used by the trading tools to value the portfolio."""
    url = f"{COINGECKO_BASE}/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    response = requests.get(url, params=params)
    data = response.json()
    return float(data.get(coin_id, {}).get("usd", 0.0))


def _daily_closes(coin_id: str, days: str) -> pd.Series:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()
    if "prices" not in data:
        return pd.Series(dtype="float64")
    return pd.Series([point[1] for point in data["prices"]])


@tool
def get_crypto_price(coin_id: str) -> str:
    """Get the current USD price for a cryptocurrency by its CoinGecko id. Example: coin_id='bitcoin'"""
    price = _price(coin_id)
    if price <= 0:
        return f"Could not fetch price for {coin_id}"
    return f"{coin_id} current price: ${price:,.2f}"


@tool
def get_price_history(coin_id: str) -> str:
    """Get the last 10 days of daily closing prices (USD) for a cryptocurrency. Example: coin_id='bitcoin'"""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "10", "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()
    if "prices" not in data:
        return f"Could not fetch price history for {coin_id}"
    prices = [
        f"    day {i + 1}: ${point[1]:,.2f}"
        for i, point in enumerate(data["prices"][-10:])
    ]
    return f"{coin_id} closing prices (last 10 days):\n" + "\n".join(prices)


@tool
def calculate_rsi(coin_id: str) -> str:
    """Calculate the current RSI (14-day) for a cryptocurrency to check if it is overbought or oversold. Example: coin_id='bitcoin'"""
    closes = _daily_closes(coin_id, "30")
    if len(closes) < 15:
        return f"Not enough data to calculate RSI for {coin_id}"
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1]
    if rsi_value > 70:
        signal = "Overbought — possible pullback ahead"
    elif rsi_value < 30:
        signal = "Oversold — possible recovery ahead"
    else:
        signal = "Neutral"
    return f"{coin_id} RSI(14): {rsi_value:.1f} → {signal}"


@tool
def get_sma(coin_id: str) -> str:
    """Get the 50-day and 200-day Simple Moving Averages (USD) to check the long-term trend. Example: coin_id='bitcoin'"""
    closes = _daily_closes(coin_id, "365")
    if len(closes) < 200:
        return f"Not enough data to calculate SMA for {coin_id}"
    sma_50 = closes.rolling(window=50).mean().iloc[-1]
    sma_200 = closes.rolling(window=200).mean().iloc[-1]
    if sma_50 > sma_200:
        cross = "Golden Cross — Bullish long-term signal"
    else:
        cross = "Death Cross — Bearish long-term signal"
    return f"{coin_id} 50-day SMA: ${sma_50:,.2f} | 200-day SMA: ${sma_200:,.2f} | {cross}"


@tool
def get_crypto_news(coin_id: str) -> str:
    """Get the latest news headlines for a cryptocurrency by its CoinGecko id. Example: coin_id='bitcoin'"""
    url = f"{COINGECKO_BASE}/news"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get("data", []) if isinstance(data, dict) else data
        headlines = [
            f"- {art.get('title')} ({art.get('source_name', 'Unknown')})"
            for art in articles[:5]
        ]
        return f"Latest news for {coin_id}:\n" + "\n".join(headlines) if headlines else f"No recent news found for {coin_id}"
    except Exception:
        return f"Could not fetch news for {coin_id}"


@tool
def place_order(coin_id: str, side: str, quantity: float) -> str:
    """Place a paper trade order for a cryptocurrency. side must be 'buy' or 'sell'. Example: coin_id='bitcoin', side='buy', quantity=0.1"""
    global _balance, _portfolio
    price = _price(coin_id)
    if price <= 0:
        return f"Could not determine price for {coin_id}"
    
    side = side.lower()
    total_cost = price * quantity
    
    if side == "buy":
        if _balance < total_cost:
            return f"Insufficient balance. Available: ${_balance:,.2f}, Required: ${total_cost:,.2f}"
        _balance -= total_cost
        if coin_id in _portfolio:
            old_qty = _portfolio[coin_id]["quantity"]
            old_cost = old_qty * _portfolio[coin_id]["avg_price"]
            new_qty = old_qty + quantity
            new_avg = (old_cost + total_cost) / new_qty
            _portfolio[coin_id] = {"quantity": new_qty, "avg_price": new_avg}
        else:
            _portfolio[coin_id] = {"quantity": quantity, "avg_price": price}
        return f"Successfully bought {quantity} {coin_id} at ${price:,.2f}. Remaining cash: ${_balance:,.2f}"
    
    elif side == "sell":
        owned = _portfolio.get(coin_id, {"quantity": 0.0})["quantity"]
        if owned < quantity:
            return f"Insufficient holdings. You own {owned} {coin_id}, tried to sell {quantity}."
        _balance += total_cost
        _portfolio[coin_id]["quantity"] -= quantity
        if _portfolio[coin_id]["quantity"] == 0:
            del _portfolio[coin_id]
        return f"Successfully sold {quantity} {coin_id} at ${price:,.2f}. Remaining cash: ${_balance:,.2f}"
    else:
        return "Invalid side. Must be 'buy' or 'sell'."


@tool
def get_stocks_owned() -> str:
    """Check what cryptocurrencies we currently own and if we are making profit or loss."""
    if not _portfolio:
        return "No cryptocurrencies currently owned in the portfolio."
    
    report = []
    for coin_id, info in _portfolio.items():
        qty = info["quantity"]
        avg_price = info["avg_price"]
        curr_price = _price(coin_id)
        current_value = qty * curr_price
        cost_basis = qty * avg_price
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        status = "PROFIT" if pnl >= 0 else "LOSS"
        report.append(f"- {coin_id}: {qty} units | Avg Buy: ${avg_price:,.2f} | Current: ${curr_price:,.2f} | PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%) [{status}]")
    return "Portfolio Holdings:\n" + "\n".join(report)


@tool
def get_balance() -> str:
    """Check how much virtual money we have left to buy crypto."""
    portfolio_value = sum(info["quantity"] * _price(coin_id) for coin_id, info in _portfolio.items())
    total_wealth = _balance + portfolio_value
    return f"Cash Balance: ${_balance:,.2f} | Portfolio Value: ${portfolio_value:,.2f} | Total Wealth: ${total_wealth:,.2f}"


api_key = os.environ["GEMINI_API_KEY"]

model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    api_key=api_key,
)

system_prompt = """You are an expert cryptocurrency trading and portfolio management analyst.

IMPORTANT: The tools use CoinGecko coin ids. Map the coin the user mentions to its id:
- Bitcoin / BTC → bitcoin
- Ethereum / ETH → ethereum
- Solana / SOL → solana
- Dogecoin / DOGE → dogecoin
- Cardano / ADA → cardano

When asked to analyze or trade a coin:
1. Gather technical data using get_crypto_price, get_price_history, calculate_rsi, get_sma, and get_crypto_news.
2. Check current portfolio and cash using get_stocks_owned and get_balance if execution or holdings are relevant.
3. Provide a clear DECISION: BUY / SELL / HOLD with CONFIDENCE: High / Medium / Low and a concise rationale.
4. If appropriate or requested, execute paper trades using place_order.

Crypto is highly volatile — always add a brief risk note."""

agent = create_agent(
    model=model,
    tools=[
        get_crypto_price,
        get_price_history,
        calculate_rsi,
        get_sma,
        get_crypto_news,
        place_order,
        get_stocks_owned,
        get_balance,
    ],
    system_prompt=system_prompt,
)

# Test the agent with sample queries
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Check my portfolio balance and analyze Bitcoin."}]}
)
print(response["messages"][-1].text)