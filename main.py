from typing import Callable, List, Optional, Tuple
from datetime import datetime, timedelta
from pymongo import MongoClient
import matplotlib.pyplot as plt
from typing import Any, Dict
import pandas as pd
import requests, os
import numpy as np
import argparse

BASE_URL = "https://financialmodelingprep.com/api/v3"

def fetch(url: str) -> Tuple[Dict[str, Any], Optional[str]]:
    headers = {'Content-Type': 'application/json'}
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        return {}, f"error fetching data from {url}, response: {res.json()}"

    data = res.json()
    return data, None


class Fmp():
    def __init__(self, token, data_dir_path="out"):
        self.token = token
        self.data_dir_path = data_dir_path
        if not os.path.exists(data_dir_path): os.makedirs(data_dir_path)
        
    def balance_sheet_path(self, ticker, period="quarter"):     
        return self.setup_dir_and_create_path(ticker, "balance_sheet", period)
    
    def income_statement_path(self, ticker, period="quarter"):  
        return self.setup_dir_and_create_path(ticker, "income_statement", period)
    
    def cash_flow_path(self, ticker, period="quarter"):         
        return self.setup_dir_and_create_path(ticker, "cash_flow", period)
    
    def intraday_prices_path(self, ticker, timeframe, date):    
        return self.setup_dir_and_create_path(ticker, f"{date}_intraday_prices", timeframe)
    
    def intraday_prices_err_path(self, ticker, timeframe, date):
        return self.setup_dir_and_create_path(ticker, f"{date}_err_intraday_prices", timeframe)
    
    
    def get_fin_statements_handler(self, path, url, ticker, logging="verbose") -> Callable:
        if os.path.exists(path):
            if logging == "verbose": print(f"    balance sheet data already exists for {ticker}")
        else:
            if logging == "verbose": print(f"    fetching balance sheet data for {ticker}")
            err = self.fetch_and_save(path, url)
            if err: print(err)
    
    
    def download_financial_statements(self, ticker, period="quarter", logging="verbose"):
        print(f" * downloading financial statements for {ticker}")
        
        # balance sheet
        url = f"{BASE_URL}/balance-sheet-statement/{ticker}?period={period}&apikey={self.token}" 
        bs_path = self.balance_sheet_path(ticker, period)
        self.get_fin_statements_handler(bs_path, url, ticker, logging=logging)

        # income statement
        url = f"{BASE_URL}/income-statement/{ticker}?period={period}&apikey={self.token}"
        income_statement_path = self.income_statement_path(ticker, period)
        self.get_fin_statements_handler(income_statement_path, url, ticker, logging=logging)
        
        # cash flow statement
        url = f"{BASE_URL}/cash-flow-statement/{ticker}?period={period}&apikey={self.token}"
        cash_flow_path = self.cash_flow_path(ticker, period)
        self.get_fin_statements_handler(cash_flow_path, url, ticker, logging=logging)


    def download_intraday_prices(self, ticker, timeframe, date) -> Optional[str]:
        """ available timeframes: 1min, 5min, 15min, 30min, 1hour """
        file_created_if_err_occured_before = self.intraday_prices_err_path(ticker, timeframe, date)
        if os.path.exists(file_created_if_err_occured_before):
            return f"no intraday prices were found before, when fetching {ticker} on {date}, skipping"

        path = self.intraday_prices_path(ticker, timeframe, date)
        if not os.path.exists(path):
            print(f"fetching intraday prices for {ticker} on {date}")
            from_date, to_date = calculate_dates(date, 4, 14)
            url = f"{BASE_URL}/historical-chart/{timeframe}/{ticker}?apikey={self.token}&serietype=line&from={from_date}&to={to_date}"
            err = self.fetch_and_save(path, url)
            if err:
                # if error inclues ""no data returned after ", then create the file_created_if_err_occured_before
                # file, so that we don't try to fetch the data again
                if "no data returned after" in err: 
                    open(file_created_if_err_occured_before, 'a').close()
                    return f"skipping {ticker} on {date}, because no data was returned before, err: {err}"

                return err

        return None

    def setup_dir_and_create_path(self, ticker, financial_statement, period):
        filename = f"{financial_statement}_{period}.parquet"
        ticker_path = f"{self.data_dir_path}/{ticker}"
        if not os.path.exists(ticker_path): os.makedirs(ticker_path)
        path = f"{ticker_path}/{filename}"
        return path
    
    def read_df(self, path, ticker) -> Tuple[pd.DataFrame, Optional[str]]:
        """ available timeframes: 1min, 5min, 15min, 30min, 1hour """
        if not os.path.exists(path): return pd.DataFrame(), f"no intraday prices data for {ticker}"
        df = pd.read_parquet(path)
        if df.empty: return pd.DataFrame(), f"no data for df at {path}"
        return df, None
    
    def read_balance_sheet(self, ticker, period="quarter") -> Tuple[pd.DataFrame, Optional[str]]:
        path = self.balance_sheet_path(ticker, period)
        return self.read_df(path, ticker)
    
    def read_income_statement(self, ticker, period="quarter") -> Tuple[pd.DataFrame, Optional[str]]:
        path = self.income_statement_path(ticker, period)
        return self.read_df(path, ticker)

    def read_cash_flow(self, ticker, period="quarter") -> Tuple[pd.DataFrame, Optional[str]]:
        path = self.cash_flow_path(ticker, period)
        return self.read_df(path, ticker)

    def read_intraday_prices(self, ticker, timeframe, date) -> Tuple[pd.DataFrame, Optional[str]]:
        path = self.intraday_prices_path(ticker, timeframe, date)
        return self.read_df(path, ticker)

    def fetch_and_save(self, path, url) -> Optional[str]:
        res, err = fetch(url)
        if err: return err
        df = pd.DataFrame(res)
        if df.empty: return f"no data returned after fetching {path}"
        df.to_parquet(path)
        return None

    def get_financial_statements(self, ticker, period="quarter") -> Tuple[pd.DataFrame, Optional[str]]:
        """ 
        Merge all 3 financial statements on the 'acceptedDate' column.
        If one of the financial statements does not exists / is empty, return an error.
        """
        income_statement_df, err = self.read_income_statement(ticker, period)
        if err: return pd.DataFrame(), err
        
        balance_sheet_df, err = self.read_balance_sheet(ticker, period)
        if err: return pd.DataFrame(), err
        
        cash_flow_df, err = self.read_cash_flow(ticker, period)
        if err: return pd.DataFrame(), err
        
        # merge the rows on the 'acceptedDate' column
        df = pd.merge(income_statement_df, balance_sheet_df, on='acceptedDate')
        df = pd.merge(df, cash_flow_df, on='acceptedDate')
        if df.empty: return pd.DataFrame(), f"no data for {ticker}"
        return df, None
        

def calculate_dates(base_date, subtract_num_days_from_t1, add_days_to_t2):
    """
    Input a date with the format of and return 2 dates with the format of YYYY-MM-DD,
    where the returned dates have a new date, defined by the input params.
    """
    date_format = "%Y-%m-%d"
    base_date_obj = datetime.strptime(base_date, date_format)
    from_date = base_date_obj - timedelta(days=subtract_num_days_from_t1)
    to_date = base_date_obj + timedelta(days=add_days_to_t2)
    
    from_date_str = datetime.strftime(from_date, date_format)
    to_date_str = datetime.strftime(to_date, date_format)
    return from_date_str, to_date_str


def calculate_price_change_x_hours_into_future(df, first_date, num_hours):
    last_date = first_date + timedelta(hours=num_hours)
    df = df[df['date'] <= last_date]    # drop all of the rows that are after the last date
    first_price = df.iloc[0]['close']   # get the first price
    last_price = df.iloc[-1]['close']   # get the last price
    price_change = last_price - first_price  # calculate the price change
    price_change_percentage = price_change / first_price * 100 # calculate the price change percentage
    msg = f"price change for {num_hours}h from {first_date} to {last_date} is {price_change_percentage:.2f}"
    
    # Find also the highest price + price change percentage
    highest_price = df['close'].max()
    highest_price_change = highest_price - first_price
    highest_price_change_percentage = highest_price_change / first_price * 100
    
    info = {
        "from": first_date,
        "to": last_date,
        "price_change": price_change,
        "price_change_percentage": price_change_percentage,
        "first_price": first_price,
        "last_price": last_price,
        'highest_price': highest_price,
        'highest_price_change_percentage': highest_price_change_percentage,
    }
    
    return price_change_percentage, info, msg


def calculate_percentage_change(df:pd.DataFrame, merge=True):
    percentage_change_data = {}  # Use a dictionary to store the percentage change columns

    # Iterate over each column in the DataFrame
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            # Calculate the percentage change for the column
            percentage_change = df[column].pct_change(-1) * 100
            # Store the calculated percentage change in the dictionary
            percentage_change_data[f"{column}_pct_change"] = percentage_change

    percentage_change_df = pd.DataFrame(percentage_change_data)
    if merge: return pd.concat([df, percentage_change_df], axis=1)
    return percentage_change_df



def download_financial_statements_and_intraday_prices(fmp: Fmp, ticker, period="quarter", logging="") -> Optional[str]:
    # download the financial statements
    fmp.download_financial_statements(ticker, period, logging=logging) 
    
    df, err = fmp.get_financial_statements(ticker, period)
    if err: 
        return f"error while merging financial statements for {ticker}: {err}"
    
    for _, financial_statements in df.iterrows():
        # if the acceptedDate column is empty, return an error
        date, err = parse_date_from_financial_statements(financial_statements['acceptedDate'])
        if err:
            print(f"error while parsing date for {ticker}: {err}")
            continue

        err = fmp.download_intraday_prices(ticker, "15min", date)
        if err:
            print(f"error fetching intraday data for {ticker} with date {date}, err: {err}")
            continue
    
    return None


def analyze_financial_statements(fmp, ticker, period="quarter", logging="") -> Tuple[Tuple[pd.DataFrame], Optional[str]]:
    df, err = fmp.get_financial_statements(ticker, period)
    if err: 
        return pd.DataFrame(), f"error while merging financial statements for {ticker}: {err}"
    
    analyzed_financial_statements = []
    
    financial_statement_pct_change = calculate_percentage_change(df, merge=True)
    for _, financial_statements in df.iterrows():
        # get the column which contains the date when the report was filed
        financial_statement_pct_change_date = financial_statements['acceptedDate']

        date, err = parse_date_from_financial_statements(financial_statement_pct_change_date)
        if err: 
            print(f"error while parsing date for {ticker}: {err}")
            continue
        
        intraday_df, err = fmp.read_intraday_prices(ticker, "15min", date)
        if err:
            print(f"error reading intraday data for {ticker} with date {date}, {err}")
            continue

        # 1. Convert all 'date' column values in intraday_df into datetime objects, 
        # assuming that they have the format of 2023-01-13 15:45:00
        intraday_df['date'] = pd.to_datetime(intraday_df['date'])
        # 2. Convert the report date string into a datetime object
        report_date = pd.to_datetime(date)
        # 3. Query for rows where the 'date' column is equal or greater than the report date
        intraday_df = intraday_df[intraday_df['date'] >= report_date]
        if intraday_df.empty:
            print(f"no intraday data for {ticker} with date {date}")
            continue

        intraday_df = intraday_df.sort_values(by='date', ascending=False)
        # get the first date
        first_date = intraday_df.iloc[0]['date']

        h24, info, msg = calculate_price_change_x_hours_into_future(intraday_df, first_date, 24)
        # if h24 is not a number, then skip this ticker
        if np.isnan(h24): continue

        # print(f"{ticker} -> {date} -> {h24:.4f}")
        # set the price change for the financial statement
        financial_statement_pct_change.loc[
            financial_statement_pct_change['acceptedDate'] == financial_statement_pct_change_date, 'price_change_h24'
        ] = h24
        
        # append the financial statement to the results
        analyzed_financial_statements.append(financial_statement_pct_change)

    return analyzed_financial_statements, None

def parse_date_from_financial_statements(date) -> Tuple[str, bool, Optional[str]]:
    """ Parse yyyy-mm-dd from the date string """
    if date == "": 
        return "", "date string is empty"

    # convert `2020-02-21 16:28:23` to `2020-02-21`
    date = date.split(" ")[0]
    if date == "" or len(date.split("-")) != 3:
        return "", f"invalid date string found: {date}"
    
    return date, None
    

# if the local file which holds the sp500 tickers does not exist, then fetch the 
# tickers from wikipedia. Else, read the tickers from the file and return them.
def get_sp500_tickers() -> Tuple[List[str], Optional[str]]:
    file_path = "./data/sp500_tickers.json"
    
    if not os.path.exists(file_path):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        
        # if an error occured while fetching the tables, return an empty list
        if len(tables) == 0:
            return [], "no tables were found on the wikipedia page"
        
        sp500 = tables[0]
        sp500.to_json(file_path)
        return sp500['Symbol'].to_list(), None
    
    sp500 = pd.read_json(file_path)
    return sp500['Symbol'].to_list(), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True, help='FMP API token')
    parser.add_argument('--num_tickers', type=int, default=500 ,required=False, help='Number of tickers to analyze')
    parser.add_argument('--mode', type=str, default="CORR" ,required=False, help='CORR to analyze the data, FETCH to fetch the data')
    args = parser.parse_args()
    
    tickers, err = get_sp500_tickers()
    if err: 
        print(err)
        return
    
    tickers = tickers[:args.num_tickers] # get the first x tickers
    token = args.token
    mode = args.mode
    
    period = "annual" # set to annual because the free api version allows only this type
    fmp = Fmp(token)
    
    if len(token) < 30:
        print("please provide a valid token")
        return
    
    if mode == "FETCH":
        for ticker in tickers: 
            err = download_financial_statements_and_intraday_prices(fmp, ticker, period, logging="")
            if err: 
                print(err)
                continue

    if mode == "CORR":
        anaylzed_financial_statements = []
        for ticker in tickers:
            res, err = analyze_financial_statements(fmp, ticker, period)
            if err: 
                print(err)
                continue
            # For each value in the array of dfs returned from the analyze_financial_statements function,
            # append the df to the anaylzed_financial_statements array
            for _, result in enumerate(res): anaylzed_financial_statements.append(result)

        print(f'number of results: {len(anaylzed_financial_statements)}')
        
        # convert the results to a dataframe
        df = pd.concat(anaylzed_financial_statements)
        # filter out columns which are not numbers
        df = df.select_dtypes(include=['float64', 'int64'])
        # see the correlation between the price change and the financial statement columns
        corr = df.corr(method='pearson')["price_change_h24"]
        # sort the correlation values
        corr = corr.sort_values(ascending=False)
        print(corr)
        
        # save the results to a file 
        df.to_csv("./data/results.csv")
        # save the correlation to a file
        corr.to_csv("./data/correlation.csv")

main()