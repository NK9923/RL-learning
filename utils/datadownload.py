import yfinance as yf 
import pandas as pd
import os
import yaml

def load_config(config_file_path="config/config.yml"):
    with open(os.path.join(os.getcwd(), config_file_path), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        COLUMNS = config.get('Input_Type')    
        return COLUMNS              

# == Download stock data from yahoo finance =======
def getData(tickers) -> list[dict, str]:
    train_data = {}
    for ticker in tickers:
        data = yf.download(ticker)
        returns = data['Adj Close'].pct_change()[1:]
        train_data[ticker] = pd.merge(data['Adj Close'], returns, on='Date', how='outer', suffixes=('_price', '_return'))
        train_data[ticker].rename(columns={'Adj Close_price': 'ClosePrice', 'Adj Close_return': 'return'}, inplace=True)
        train_data[ticker] = train_data[ticker].iloc[1:]

    return train_data        

# == Use Future Data ==============================
def getData_from_csv(tickers, full_data = False) -> list[dict, str]:
    train_data = {}
    problem_ticker = []
    for Ticker in tickers: 
        file_path = os.path.join("Input", f"{Ticker}.csv") 
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df.index =  pd.to_datetime(df.index)
            if not full_data:    
                columns = load_config()                       
                df['MA_volume_100_days'] = df['Volume'].rolling(window=100).mean()   
                df['MA_volume_100_days'] = df['MA_volume_100_days'].fillna(method='bfill')   
                df['Adjusted_Close_EUR'] = df['AdjustedClose'] * df['Cross_Rate'] 
                df['ContractValue_EUR']  = df['ContractValue'] * df['Cross_Rate']                      
                train_data[Ticker] = df[columns]    
            else:
                train_data[Ticker] = df                                                        
        else:
            problem_ticker.append(Ticker)
    return train_data, problem_ticker

def getData_from_csv1(tickers) -> list[dict, str]:
    problem_ticker = []
    for Ticker in tickers: 
        file_path = os.path.join("Input", f"{Ticker}.csv") 
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df.index =  pd.to_datetime(df.index)
            T = df[['Close']]     
            T = T.rename(columns={'Close': 'AdjustedClose'}, inplace=False)                                        
        else:
            problem_ticker.append(Ticker)
    return T
