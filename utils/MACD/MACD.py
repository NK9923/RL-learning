from os import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
import matplotlib.ticker as ticker

from utils.Configure import *
from utils.datadownload import *

class MACDStrategy:
    def __init__(self, envir, config, slow, fast, smooth, Current_market):
        self.Environement = envir     
        self.Config = config          
        self.prices = self.Config.test_data['Adjusted_Close_EUR']                
        self.slow = slow
        self.fast = fast
        self.smooth = smooth
        self.market = Current_market     
        self.MACD_porfolio_value = 0           

        self.output_dir = os.path.join(self.Config.Output_PATH, "MACD_strategy")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)           
        
        self.actions = []        
        evaluate = self.run_strategy() 

        if evaluate:
            self.Risk_measures_MACD = self._eval_MACD_future_strategy(evaluate)                  
                     
 
    def _get_macd(self) -> pd.DataFrame:
        exp1 = self.prices.ewm(span=self.fast, adjust=False).mean()
        exp2 = self.prices.ewm(span=self.slow, adjust=False).mean()
        macd = pd.DataFrame(exp1 - exp2)
        macd.columns = ['macd']               
        signal = pd.DataFrame(macd['macd'].ewm(span=self.smooth, adjust=False).mean())
        signal.columns = ['signal']        
        hist = pd.DataFrame(macd['macd'] - signal['signal'], columns=['hist'])
        return pd.concat([macd, signal, hist], axis=1)


    def _implement_macd_strategy(self, data) ->  tuple[list, list, list]: 
        buy_price = []
        sell_price = []
        macd_signal = []
        signal = 0
        
        for i in range(len(data)):
            if data['macd'][i] > data['signal'][i]:
                if signal != 1:
                    buy_price.append(self.prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)
            elif data['macd'][i] < data['signal'][i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(self.prices[i])
                    signal = -1
                    macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    macd_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        return buy_price, sell_price, macd_signal


    def _plotMACD(self, buy_price, sell_price, macd_stock, portfolio_development) -> None:   
                 
        def mio_formatter(x, pos):
                return f'{x / 1e6:.2f} Mio' 
        
        fig = plt.figure(figsize=(10, 10))    
        ax0 = plt.subplot2grid((9, 1), (0, 0), rowspan=3, colspan=1)
        ax1 = plt.subplot2grid((9, 1), (3, 0), rowspan=3, colspan=1)
        ax2 = plt.subplot2grid((9, 1), (6, 0), rowspan=3, colspan=1)
         
        ax0.plot(self.prices.index[0:len(portfolio_development)], portfolio_development, color='steelblue', label = "Portfolio developement MACD strategy")
        formatter = ticker.FuncFormatter(mio_formatter)
        ax0.yaxis.set_major_formatter(formatter)           
        ax0.axhline(y=portfolio_development[0], color='tomato', linestyle='--', linewidth=0.75)   
        ax0.set_title('MACD strategy\n({})'.format(self.Config.test_data["Name"][0]),fontsize=14, fontweight='bold')  
        ax0.tick_params(axis='x', labelsize=8)
        ax0.legend()
        ax0.grid()                   
          
        ax1.plot(self.prices, color = 'black', linewidth = 1.4)
        ax1.plot(self.prices.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
        ax1.plot(self.prices.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
        ax1.tick_params(axis='x', labelsize=8)        
        ax1.legend()
                 
        ax2.plot(macd_stock['macd'], color = 'royalblue', linewidth = 1.5, label = 'MACD-line')
        ax2.plot(macd_stock['signal'], color = 'orange', linewidth = 1.5, label = 'signal-line')
        ax2.tick_params(axis='x', labelsize=8)            

        for i in range(len(macd_stock)):
            if str(macd_stock['hist'][i])[0] == '-':
                ax2.bar(macd_stock.index[i], macd_stock['hist'][i], color = '#ef5350')
            else:
                ax2.bar(macd_stock.index[i], macd_stock['hist'][i], color = '#26a69a')
        
        plt.legend(loc = 'lower right')
        plt.subplots_adjust(hspace=0.5)
        output_filename = "{}_MACD.pdf".format(self.market)
        output_path = os.path.join(self.output_dir, output_filename)

        plt.savefig(output_path, format='pdf')
        plt.close()
     
           
    def run_strategy(self) ->  tuple[list, list, list]: 
        try:
            macd_stock = self._get_macd()
            buy_price, sell_price, macd_signal = self._implement_macd_strategy(macd_stock)
            self.actions = [x + 1 for x in macd_signal]        

            #self._plotMACD(buy_price, sell_price, macd_stock)
            return (buy_price, sell_price, macd_signal, macd_stock)  
                     
        except Exception as e:
             print(f"Running MACD strategy caused an exception: {e}")             
             return None                        


    def _eval_MACD_future_strategy(self, strategy) -> dict:         
        try:   
            done = False    
            i = 0      
                          
            while(not done):
                _, done = self.Environement.step_eval(action=self.actions[i], debug = False)  
                i += 1                              

            self.MACD_porfolio_value = self.Environement.CalcReturnSeries() 
            buy_price, sell_price, macd_signal, macd_stock = strategy
            self._plotMACD(buy_price, sell_price, macd_stock, portfolio_development = self.Environement.portfolio_history)   
            
            return self.Environement.risk_measures      
        except Exception as e:
             print(f"Could not evaluate MACD strategy: {e}")    
      
                    
if __name__ == "__main__":   
    
    models = '20230906-160644' 
    Asset = 'CC'
    cfg = Config(create_output_dir = models)
    cfg.update_settings('Commodities')
    cfg.getTrainAndTestData(Asset) 
    env, agent = env_agent_config(cfg.test_data['RoR'].tolist(), cfg, 'test', individual_ticker = Asset)  

    MACD = MACDStrategy(envir = env, config=cfg, slow = 26, fast = 12, smooth = 9, Current_market=Asset)

    MACD.Risk_measures_MACD  