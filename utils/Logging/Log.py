import os
import logging
import datetime as dt
from pickle import NONE

def mio_formatter(x):
        return f'{x / 1e6:.2f} Mio'   

def FileLogging(cfg, key, data = None, content = None):
    log_file = cfg.Output_PATH + '\\logs'
    
    if not os.path.exists(log_file):
        os.makedirs(log_file)           
    
    current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file + '\\' + key + "_test.log", mode='w')
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    if content is None and data is not None:

        content = f"""
        ------------------------------------------------------------------------------------------
        |{'Initial captial':<30}|{mio_formatter(data['IC']):<57}|            
        |{'Net worth':<30}|{round(data['abs_g'] + data['IC'], 4):<20} Max net worth: {round(data['MaxVal'], 2): <20} |
        |{'         ':<30}|{'':<20} Min net worth: {round(data['MinVal'], 2): <20} |        
        |{'Profit in Mio':<30}|{mio_formatter(round(data['abs_g'], 4)):<57}|        
        |{'Profit exact':<30}|{round(data['abs_g'], 4):<57}|
        |{'Return':<30}|{round(data['abs_ret'] * 100, 3):<8} % {' ':<46}|
        |{'Mean annualized return':<30}|{round(data['MeanReturn'] * 100, 3):<8} % {' ':<46}|           
        |{'Volatility (annualized)':<30}|{round(data['vola'],2):<8}  {' ':<47}| 
        |{'Risk free rate':<30}|{round(data['RF'],2):<3} % {' ':<51}|   
        |{'Value at Risk (95 %)':<30}|{round(data['VaR'],2):<8} {' ':<48}| 
        |{'Conditional VaR (ES)':<30}|{round(data['CVaR'],2):<8} {' ':<48}| 
        |{'Maximum Drawdown':<30}|{round(data['DD'],2):<16} {' ':<40}|
        |{'Sortino ratio':<30}|{round(data['SR'],2):<8}{' ':<49}|
        |{'Calmar ratio':<30}|{round(data['Calm'],2):<16} {' ':<40}| 
        |{'Tail ratio':<30}|{round(data['TailRatio'],2):<8} {' ':<48}|   
        |{'Downside risk':<30}|{round(data['DownSideRisk'],2):<8} {' ':<48}|                                                  
        |{'Sharpe ratio':<30}|{round(data['Sharp'],2):<8} {' ':<48}|                                                                                   
        ------------------------------------------------------------------------------------------
        """    
        print(content)         

    logging.info(content)            
