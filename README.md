## Testing correlation between stock price change and financial reports

A small script to test if there's a correlation betweeen the change in the stock price on the day when the company releases its financial reports:

- Balance Sheet
- Income Statement
- Cash Flow Statement

To give a concrete example, this tries to test the probablility that a stock price will increase on the day when the company releases its financial reports and one of the values in these reports shows an unexpected increase / decrease since the last year.

Note that the stock price change is calculated as, `stock_price_on_day_of_release - stock_price_on_next_day_after_relase`.

### Disclaimer

Open a PR if you see a mistake from my part.

### Usage

Financial modeling prep is used to gather the data, thus you need to have an API key from them. You can get one [here](https://financialmodelingprep.com/developer/docs/).
The free version won't be able to gather all of the tickers in one run, because roughly 7 requests are needed for 1 ticker.

`num_tokens` is used to run through the script quickly, if needed.

```bash
# run at first to gather the data
python3 main.py --token YOUR_TOKEN --num_tickers 500 --mode FETCH
# run to calulate the correlation on the saved data.
python3 main.py --token YOUR_TOKEN --num_tickers 500 --mode CORR
```

### Results

All of the correlation results can be found in the `data/correlation.csv` file. 2389 years of reference were used to find these results.

The highest correlation can be found between the stock price change and otherLiabilities_pct_change (-0.15). Other than that, the correlation is not significant.
