import yfinance as yf
import pandas 
import os

data_path = os.getenv("AGENT_TRADE_PATH", "./data/")


# the top 25 as of SP500
SP_500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "TSLA", "GOOG", "BRK.B", "META", "UNH", "XOM", "LLY", "JPM", "V", 
    "PG", "MA", "AVGO", "HD", "CVX", "MRK", "ABBV", "COST", "PEP", "ADBE"
]

for SP in SP_500:
    data = yf.download(SP, start="2024-02-07", end="2025-02-07")

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save each ticker's data to a CSV file
    filename = f"{SP}.csv"  # File name for each stock
    download_file = data_path + filename
    data.to_csv(download_file, sep="\t", index=True)

