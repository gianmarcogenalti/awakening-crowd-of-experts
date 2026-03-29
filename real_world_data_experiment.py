import yaml
from core import run_reproducible_campaign

try:
    with open('finance_tickers.yml', 'r') as file:
            tickers = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: Could not find finance_tickers.yml. Please ensure the file exists.")

run_reproducible_campaign(tickers)
