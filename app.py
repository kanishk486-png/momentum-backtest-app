import datetime
import warnings
import backtrader as bt
import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Suppress minor warnings for clean console outputs
warnings.filterwarnings('ignore')

# --- 1. Define Momentum Strategy with Trailing Stop-Loss ---
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
        
        # Entry Logic: Monthly rebalancing + positive momentum filter
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
        # Exit Logic: Dynamic Trailing Stop-Loss tracking
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

# --- 2. Streamlit Web App Layout ---
def main():
    st.set_page_config(page_title="Momentum Backtest Dashboard", layout="wide")
    
    st.title("📈 Momentum Strategy & Trailing Stop-Loss Dashboard")
    st.markdown("Configure your backtest parameters in the sidebar and simulate trading performance instantly.")

    # --- Sidebar Inputs ---
    st.sidebar.header("Strategy Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="BEL.NS")
    start_date = st.sidebar.date_input("Start Date", value=datetime.date(2021, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.date(2026, 1, 1))
    
    initial_cash = st.sidebar.number_input("Initial Capital (₹)", value=100000.0, step=10000.0)
    trail_pct = st.sidebar.slider("Trailing Stop-Loss (%)", min_value=1.0, max_value=20.0, value=8.0, step=0.5) / 100.0
    momentum_period = st.sidebar.number_input("Momentum Lookback (Days)", value=252, step=10)
    rebalance_days = st.sidebar.number_input("Rebalance Frequency (Days)", value=21, step=1)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Downloading historical data and running simulation..."):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.error("Failed to fetch data. Please verify your ticker symbol and date range.")
                return

            cerebro = bt.Cerebro()
            cerebro.addstrategy(
                MomentumTrailingSLStrategy,
                momentum_period=int(momentum_period),
                trail_percent=trail_pct,
                rebalance_days=int(rebalance_days)
            )

            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)

            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
            
            results = cerebro.run()
            strat = results[0]
            
            final_value = cerebro.broker.getvalue()
            total_return = ((final_value - initial_cash) / initial_cash) * 100

            # --- Display Performance Metrics ---
            st.success("Backtest Completed Successfully!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Starting Portfolio Value", f"₹{initial_cash:,.2f}")
            col2.metric("Final Portfolio Value", f"₹{final_value:,.2f}")
            col3.metric("Total Return", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")

            # --- Trade Execution Log ---
            st.subheader("📋 Trade Execution Log")
            if strat.trade_logs:
                logs_df = pd.DataFrame(strat.trade_logs)
                st.dataframe(logs_df, use_container_width=True)
            else:
                st.info("No trades were executed during this period based on current criteria.")

            # --- Equity Curve Plotting ---
            st.subheader("📊 Portfolio Equity Growth Curve")
            returns_dict = strat.analyzers.timereturn.get_analysis()
            returns_df = pd.DataFrame(list(returns_dict.items()), columns=['Date', 'Return'])
            returns_df['Date'] = pd.to_datetime(returns_df['Date'])
            returns_df.set_index('Date', inplace=True)
            
            equity_curve = initial_cash * (1 + returns_df['Return']).cumprod()
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(equity_curve.index, equity_curve.values, label="Portfolio Equity", color="#1f77b4", linewidth=2)
            ax.set_title(f"Backtest Performance: {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value (₹)")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)

if __name__ == "__main__":
    main()