import datetime
import warnings
import backtrader as bt
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress minor warnings
warnings.filterwarnings('ignore')

# --- 1. Define Global Stock Universe ---
GLOBAL_UNIVERSE = {
    "US Tech & Growth": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "Global & European Bluechips": ["ASML.AS", "SAP.DE", "MC.PA", "NESN.SW", "NVO"],
    "Indian Market Leaders": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "BEL.NS"]
}

# Flatten for selection
ALL_TICKERS = [ticker for category in GLOBAL_UNIVERSE.values() for ticker in category]

# --- 2. Define Momentum Strategy with Trailing Stop-Loss ---
class MomentumTrailingSLStrategy(bt.Strategy):
    params = (
        ("momentum_period", 252),
        ("trail_percent", 0.08),
        ("rebalance_days", 21),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.high = self.datas[0].high
        self.roc = bt.indicators.RateOfChange(self.dataclose, period=self.params.momentum_period)
        self.highest_price = 0.0
        self.stop_loss_price = 0.0
        self.day_counter = 0
        self.trade_logs = []

    def next(self):
        self.day_counter += 1
        if not self.position:
            if self.day_counter % self.params.rebalance_days == 0:
                if self.roc[0] > 0:
                    size = int((self.broker.getcash() * 0.95) / self.dataclose[0])
                    if size > 0:
                        self.buy(size=size)
                        self.highest_price = self.high[0]
                        self.stop_loss_price = self.highest_price * (1 - self.params.trail_percent)
                        self.trade_logs.append({
                            "Date": self.datas[0].datetime.date(0),
                            "Action": "BUY",
                            "Price": round(self.dataclose[0], 2),
                            "Quantity": size
                        })
        else:
            if self.high[0] > self.highest_price:
                self.highest_price = self.high[0]
                self.stop_loss_price = self.highest_price * (1 - self.params.trail_percent)

            if self.dataclose[0] <= self.stop_loss_price:
                self.trade_logs.append({
                    "Date": self.datas[0].datetime.date(0),
                    "Action": "STOP-LOSS SELL",
                    "Price": round(self.dataclose[0], 2),
                    "Quantity": self.position.size
                })
                self.close()
                self.highest_price = 0.0
                self.stop_loss_price = 0.0

# --- 3. Global Momentum Scanner Helper ---
@st.cache_data(ttl=3600)
def scan_global_momentum(tickers, start_date, end_date):
    """
    Downloads historical data for a list of tickers and ranks them by 12-month momentum (Rate of Change).
    """
    performance_data = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty and len(df) > 50:
                # Flatten multi-index if present
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(ticker, level=1, axis=1) if ticker in df.columns.levels[1] else df.droplevel(1, axis=1)
                
                initial_price = df['Close'].iloc[0]
                current_price = df['Close'].iloc[-1]
                pct_gain = ((current_price - initial_price) / initial_price) * 100
                
                performance_data.append({
                    "Ticker": ticker,
                    "Total Gain (%)": round(pct_gain, 2),
                    "Current Price": round(float(current_price), 2)
                })
        except Exception:
            continue
            
    if performance_data:
        df_res = pd.DataFrame(performance_data)
        return df_res.sort_values(by="Total Gain (%)", ascending=False).reset_index(drop=True)
    return pd.DataFrame()

# --- 4. Monte Carlo Forecasting Engine ---
def run_monte_carlo_forecast(df, ticker, days_to_forecast=252, num_simulations=1000):
    log_returns = np.log(1 + df['Close'].pct_change()).dropna()
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    last_price = float(df['Close'].iloc[-1])
    
    np.random.seed(42)
    Z = np.random.standard_normal((days_to_forecast, num_simulations))
    daily_returns = np.exp(drift + stdev * Z)
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    for t in range(1, days_to_forecast):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths, last_price

# --- 5. Streamlit Web App Layout ---
def main():
    st.set_page_config(page_title="Global Momentum & Profitability Scanner", layout="wide")
    
    st.title("🌍 Global Stock Momentum & Profit Optimization Dashboard")
    st.markdown("Scan world markets across US, European, and Indian exchanges to identify top momentum leaders, backtest strategies, and forecast profit probabilities.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Global Universe & Strategy Settings")
    
    selected_category = st.sidebar.selectbox("Market Category", list(GLOBAL_UNIVERSE.keys()))
    tickers_to_scan = GLOBAL_UNIVERSE[selected_category]
    
    selected_ticker = st.sidebar.selectbox("Select Asset to Backtest / Forecast", tickers_to_scan)
    
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2021, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date(2026, 1, 1))
    
    initial_cash = st.sidebar.number_input("Initial Capital (Currency Units)", value=100000.0, step=10000.0)
    trail_pct = st.sidebar.slider("Trailing Stop-Loss (%)", min_value=1.0, max_value=20.0, value=8.0, step=0.5) / 100.0
    
    st.sidebar.markdown("---")
    forecast_days = st.sidebar.slider("Monte Carlo Forecast Horizon (Days)", min_value=30, max_value=252, value=126, step=10)
    target_profit_pct = st.sidebar.slider("Target Profit Threshold (%)", min_value=5, max_value=100, value=20, step=5) / 100.0

    # --- Main Action Tabs ---
    tab1, tab2, tab3 = st.tabs(["🌐 Global Momentum Scanner", "📊 Backtest Performance", "🔮 Probabilistic Risk & Profit Forecast"])

    with tab1:
        st.subheader(f"Momentum Rankings: {selected_category}")
        st.markdown("Scanning historical performance across this regional basket to identify current market leaders:")
        
        if st.button("Run Global Momentum Scan", type="primary"):
            with st.spinner("Fetching global market data and calculating momentum..."):
                ranked_df = scan_global_momentum(tickers_to_scan, start_date, end_date)
                if not ranked_df.empty:
                    st.dataframe(ranked_df, use_container_width=True)
                    st.success("Scan complete! Select the top-ranking asset from the sidebar to run backtests and risk forecasts.")
                else:
                    st.warning("Could not retrieve data for this category.")

    # Download data for selected ticker for Tabs 2 & 3
    df = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df = df.xs(selected_ticker, level=1, axis=1) if selected_ticker in df.columns.levels[1] else df.droplevel(1, axis=1)

    with tab2:
        st.subheader(f"Backtest Results for: {selected_ticker}")
        if df.empty:
            st.error("Please ensure valid data is available for the selected asset.")
        else:
            if st.button("Run Backtest Simulation"):
                with st.spinner(f"Running backtest strategy for {selected_ticker}..."):
                    cerebro = bt.Cerebro()
                    cerebro.addstrategy(MomentumTrailingSLStrategy, trail_percent=trail_pct)
                    cerebro.adddata(bt.feeds.PandasData(dataname=df))
                    cerebro.broker.setcash(initial_cash)
                    cerebro.broker.setcommission(commission=0.001)
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
                    
                    results = cerebro.run()
                    strat = results[0]
                    final_value = cerebro.broker.getvalue()
                    total_return = ((final_value - initial_cash) / initial_cash) * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Starting Capital", f"{initial_cash:,.2f}")
                    col2.metric("Final Value", f"{final_value:,.2f}")
                    col3.metric("Total Return", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")

                    st.subheader("📋 Trade Log")
                    if strat.trade_logs:
                        st.dataframe(pd.DataFrame(strat.trade_logs), use_container_width=True)
                    else:
                        st.info("No trades executed for this asset during the timeframe.")

                    # Equity Curve
                    returns_dict = strat.analyzers.timereturn.get_analysis()
                    returns_df = pd.DataFrame(list(returns_dict.items()), columns=['Date', 'Return'])
                    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
                    returns_df.set_index('Date', inplace=True)
                    equity_curve = initial_cash * (1 + returns_df['Return']).cumprod()
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(equity_curve.index, equity_curve.values, color="#2ca02c", linewidth=2)
                    ax.set_title(f"Portfolio Equity Curve ({selected_ticker})")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    st.pyplot(fig)

    with tab3:
        st.subheader(f"Probabilistic Profit & Risk Analysis: {selected_ticker}")
        if not df.empty:
            price_paths, current_price = run_monte_carlo_forecast(df, selected_ticker, days_to_forecast=forecast_days)
            terminal_prices = price_paths[-1]
            terminal_returns = (terminal_prices - current_price) / current_price
            
            prob_profit = np.mean(terminal_returns > 0) * 100
            prob_target = np.mean(terminal_returns >= target_profit_pct) * 100
            var_95 = np.percentile(terminal_returns, 5) * 100
            expected_return = np.mean(terminal_returns) * 100

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Probability of Profit", f"{prob_profit:.1f}%")
            r2.metric(f"Prob. of Target (≥{int(target_profit_pct*100)}%)", f"{prob_target:.1f}%")
            r3.metric("Expected Return", f"{expected_return:.1f}%")
            r4.metric("Value at Risk (VaR 95%)", f"{var_95:.1f}%", delta_color="inverse")

            st.markdown("---")
            st.subheader("Monte Carlo Future Price Simulation Paths")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(price_paths[:, :40], color='skyblue', alpha=0.4)
            ax2.plot(price_paths.mean(axis=1), color='darkorange', linewidth=2, label='Expected Path (Mean)')
            ax2.axhline(y=current_price, color='black', linestyle='--', label=f'Current Price ({current_price:.2f})')
            ax2.set_title(f"Monte Carlo Projections for {selected_ticker}")
            ax2.set_xlabel("Trading Days Ahead")
            ax2.set_ylabel("Price")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
