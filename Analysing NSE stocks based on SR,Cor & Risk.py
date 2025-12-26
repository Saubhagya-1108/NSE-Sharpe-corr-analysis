#Analysing NSE Stocks based on AVG cor<0.40,SR>1,RISK<25% with portfolio return 18.23% 
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Define Time Periods 
end_date = dt.datetime(2025,12,25)
startdate = end_date - dt.timedelta(days=365*5) 

training_end = end_date - dt.timedelta(days=365*2)
training_start = startdate  
test_start = training_end
test_end = end_date

# Downloading Data
stocks = ['^NSEI', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS',
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS', 'LTIM.NS',
    'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS',
    'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'BAJAJ-AUTO.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'APOLLOHOSP.NS',
    'BHARTIARTL.NS',
    'ULTRACEMCO.NS', 'GRASIM.NS',
    'TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 
    'ADANIENT.NS', 'ADANIPORTS.NS',
    'POWERGRID.NS', 'NTPC.NS',
    'LT.NS', 'TITAN.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 
    'ASIANPAINT.NS', 'DIVISLAB.NS', 'TRENT.NS']

print("\nDownloading Data...")
df = yf.download(stocks, start=startdate, end=end_date, group_by="ticker", progress=False)
adj_close_price = df.xs('Close', level=1, axis=1).dropna(axis=1)

#  Training Metrics Calculation 
training_prices = adj_close_price[(adj_close_price.index >= training_start) & (adj_close_price.index < training_end)]
training_returns = training_prices.pct_change().dropna()

annual_returns_training = training_returns.mean() * 252
annual_volatility_training = training_returns.std() * np.sqrt(252)
sharpe_ratio_training = annual_returns_training / annual_volatility_training

#  Selection Logic with Correlation Filter 
eligible_by_vol = sharpe_ratio_training[annual_volatility_training <= 0.25]
top_10_names = eligible_by_vol.nlargest(10).index.tolist()

# Calculate Average Correlation for these 10 candidates
corr_matrix_pre = training_returns[top_10_names].corr()
avg_corr_series = (corr_matrix_pre.sum() - 1) / (len(top_10_names) - 1)

# Filter: Avg Corr <= 0.40 and Sharpe >= 1.0
final_portfolio_stocks = [
    s for s in top_10_names 
    if avg_corr_series[s] <= 0.40 and sharpe_ratio_training[s] >= 1.0
]

print("\n" + "=" * 70)
print("FINAL PORTFOLIO SELECTION (Vol <= 25%, Avg Corr <= 0.40, Sharpe >= 1.0)")
print("=" * 70)
for i, stock in enumerate(final_portfolio_stocks, 1):
    print(f"{i:2d}. {stock:15s} | Sharpe: {sharpe_ratio_training[stock]:.3f} | Avg Corr: {avg_corr_series[stock]:.3f}")
print("=" * 70)

#  Correlation Visualizations 
if final_portfolio_stocks:
    final_corr = training_returns[final_portfolio_stocks].corr()
    final_avg_corr = (final_corr.sum() - 1) / (len(final_portfolio_stocks) - 1)

    fig, (ax_corr1, ax_corr2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Heatmap (Pair-wise)
    sns.heatmap(final_corr, annot=True, cmap='RdYlGn', center=0, fmt=".2f", linewidths=0.5, ax=ax_corr1)
    ax_corr1.set_title('Pair-wise Correlation Matrix', fontweight='bold', fontsize=14)

    # 2. Vertical Bar Graph (Average Correlation vs Threshold)
    bars = ax_corr2.bar(final_avg_corr.index, final_avg_corr.values, color='skyblue', edgecolor='navy', alpha=0.7)
    ax_corr2.axhline(y=0.40, color='red', linestyle='--', linewidth=2, label='0.40 Maximum Threshold')
    
    ax_corr2.set_title('Average Correlation per Selected Stock', fontweight='bold', fontsize=14)
    ax_corr2.set_ylabel('Avg Correlation Coefficient')
    ax_corr2.set_xticklabels(final_avg_corr.index, rotation=45, ha='right')
    ax_corr2.grid(axis='y', linestyle=':', alpha=0.6)
    ax_corr2.legend()
    
    # Adding value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax_corr2.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

#  Backtesting & Terminal Output 
test_prices = adj_close_price[(adj_close_price.index >= test_start) & (adj_close_price.index <= test_end)]
test_returns = test_prices[final_portfolio_stocks].pct_change().dropna()
test_years = (test_end - test_start).days / 365.25

weights = np.array([1/len(final_portfolio_stocks)] * len(final_portfolio_stocks))
portfolio_daily_returns = (test_returns * weights).sum(axis=1)
portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
individual_cumulative = (1 + test_returns).cumprod()

port_cagr = (portfolio_cumulative.iloc[-1]**(1/test_years)) - 1
port_vol = portfolio_daily_returns.std() * np.sqrt(252)
port_sharpe = port_cagr / port_vol

print("\n" + "=" * 90)
print(f"{'STOCK (TESTING PERIOD)':<35} | {'CAGR (%)':<12} | {'RISK (%)':<12} | {'SHARPE':<8}")
print("-" * 90)
for s in final_portfolio_stocks:
    s_cagr = (individual_cumulative[s].iloc[-1]**(1/test_years)) - 1
    s_vol = test_returns[s].std() * np.sqrt(252)
    s_sha = s_cagr / s_vol
    print(f"{s:<35} | {s_cagr*100:10.2f}% | {s_vol*100:10.2f}% | {s_sha:8.3f}")
print("-" * 90)
print(f"{'*** TOTAL PORTFOLIO ***':<35} | {port_cagr*100:10.2f}% | {port_vol*100:10.2f}% | {port_sharpe:8.3f}")
print("=" * 90)

#  Performance Visualizations 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.plot(portfolio_cumulative.index, (portfolio_cumulative - 1) * 100, 
          linewidth=4, label='PORTFOLIO', color='gold', zorder=10)
for stock in final_portfolio_stocks:
    ax1.plot(individual_cumulative.index, (individual_cumulative[stock] - 1) * 100, alpha=0.4, label=stock)
ax1.set_title('Test Period: Cumulative Growth (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Total Return (%)')
ax1.legend(loc='upper left', fontsize='small', ncol=2)
ax1.grid(True, alpha=0.2)

plot_data = pd.Series({s: ((individual_cumulative[s].iloc[-1]**(1/test_years)) - 1) * 100 for s in final_portfolio_stocks})
plot_data['PORTFOLIO'] = port_cagr * 100
plot_data = plot_data.sort_values()

colors = ['gold' if x == 'PORTFOLIO' else 'skyblue' for x in plot_data.index]
ax2.barh(plot_data.index, plot_data.values, color=colors)
ax2.set_title('Test Period: CAGR (%) Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('CAGR (%)')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()

plt.show()
