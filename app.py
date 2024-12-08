import pandas as pd
import yfinance as yf 
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st

st.title("Investment Portfolio Dashboard")
st.subheader("Compare your portfolio to the S&P 500")

# User input for assets
assets = st.text_input("Provide your assets (comma separated)", "AAPL, MSFT, GOOGL, AMZN, BABA")
asset_list = [x.strip() for x in assets.split(',')]

# User input for start date
start = st.date_input(
    "Pick a start date!",
    value=pd.to_datetime("2020-01-01")
)

# User input for weights
default_weight = ",".join([f"{1/len(asset_list):.2f}"] * len(asset_list))
weights_input = st.text_input(
    "Provide weights for each asset (comma separated, must sum to 1)",
    default_weight
)

try:
    # Convert weights input to a NumPy array
    weights = np.array([float(x.strip()) for x in weights_input.split(',')])
    
    # Validate weights
    if len(weights) != len(asset_list):
        st.error("Number of weights must match number of assets!")
    elif not np.isclose(np.sum(weights), 1.0):
        st.error("Weights must sum to 1!")
    else:
        # Download asset data
        data = yf.download(asset_list, start=start)['Adj Close']
        
        # Handle the case when a single asset is provided
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Calculate daily returns
        ret_df = data.pct_change().dropna()
        
        # Calculate cumulative returns for the portfolio using custom weights
        portfolio_returns = (ret_df * weights).sum(axis=1)
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Download S&P 500 data
        SP500 = yf.download("^GSPC", start=start)['Adj Close']
        SP500_returns = SP500.pct_change().dropna()
        cumulative_SP500 = (1 + SP500_returns).cumprod() - 1

        # Align the indices
        combined_returns = pd.concat([cumulative_SP500, cumulative_portfolio_returns], axis=1).dropna()
        combined_returns.columns = ["S&P 500", "Portfolio"]
        
        # Calculate portfolio risk (standard deviation)
        portfolio_std = (weights @ ret_df.cov() @ weights) ** 0.5
        
        # Calculate benchmark risk (S&P 500 standard deviation)
        benchmark_std = SP500_returns.std().item()
        
        # Display Portfolio vs S&P 500
        st.subheader("Portfolio vs Index")
        st.line_chart(combined_returns)
        
        # Display Risk Metrics
        st.subheader("Portfolio Risk:")
        st.write(f"Standard Deviation: {portfolio_std:.4f}")
        
        st.subheader("Benchmark Risk:")
        st.write(f"S&P 500 Standard Deviation: {benchmark_std:.4f}")
        
        # Display Portfolio Composition
        st.subheader("Portfolio Composition:")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#28282B')
        ax.pie(weights, labels=asset_list, autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_facecolor('#28282B')
        plt.tight_layout()
        st.pyplot(fig)

except ValueError:
    st.error("Please enter valid numerical weights separated by commas!") 