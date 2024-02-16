import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import pandas as pd
from datadownload import *
import numpy as np
import re
import random
import math

def parse_log_file(log_file_path, pattern) -> list:
    values  = []
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = re.search(pattern, line)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError:
                    value = match.group(1)
                values.append(value)
    return values


def parseLogFile(log_file_path, pattern):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    step_pattern = re.compile(r"Step: (\d+)")
    corresponds_pattern = re.compile(r"Corresponds to: (\d{4}-\d{2}-\d{2})")
    #buy_pattern = re.compile(pattern)
    buy_pattern = re.compile(r"(?:Revised action! maximum Exposure reached, action was changed to: )?" + pattern)    
    #price_pattern = re.compile(r"\$([\d.]+)")

    current_step = {}
    steps = []

    lines = log_content.splitlines()
    for i in range(len(lines)):
        line = lines[i]
        step_match = step_pattern.search(line)
        corresponds_match = corresponds_pattern.search(line)

        if step_match and corresponds_match:
            next_line = lines[i + 1]
            buy_match = buy_pattern.search(next_line)
            #price_match = price_pattern.search(next_line)
            if buy_match: #and price_match:
                current_step['Date'] = corresponds_match.group(1)
                current_step['Decision'] = str(buy_match.group(0))
                #current_step['Price'] = float(price_match.group(1))
                steps.append(current_step.copy())
                current_step.clear()

    return steps


def getPerformance(ticker, *args, pattern=None):
    cfg, env = args
    NN_profits = []
    for key in ticker:
        if cfg.algo_name == "DQN":
            log_file_path = "logs//DQN//DQN_test" + key + ".log"
        elif cfg.algo_name == "DDQN":
             log_file_path = "logs//DDQN//DDQN_test" + key + ".log"

        if pattern is None:
            profits = parse_log_file(log_file_path, r'Absolute gain:\s+\$\s*(-?\d+(?:\.\d{2})?)')[0]
        else:
            profits = parse_log_file(log_file_path, r'Absolute gain:\s+\$\s*(-?\d+(?:\.\d{2})?)')[pattern]
        NN_profits.append(profits)
    return NN_profits


def equally_weighted_PF(portfolio_developement, initial_capital) -> list:
    portfolio_developments_series = {key: pd.Series(values) for key, values in portfolio_developement.items()}    
    log_returns = {key: np.log(portfolio_developments_series[key] / portfolio_developments_series[key].shift(1)) for key in portfolio_developments_series}  
    log_returns_df = pd.DataFrame.from_dict(log_returns)
    log_returns_df = log_returns_df.dropna(how='any')    
    log_returns_sum = log_returns_df.sum(axis=1)
    log_returns_sum = log_returns_sum.div(len(log_returns_df.columns))

    equally_weighted_portfolio = (1 + log_returns_sum).cumprod()
    equally_weighted_portfolio = equally_weighted_portfolio *  initial_capital   
    return equally_weighted_portfolio.tolist()    
      
   
class Plotting:
    def __init__(self, cfg, PATH, portfolio_developement, data, test_data, asset_Class, settings, Dates):       
        self.data = test_data
        self.Config = cfg        
        self.testing_PV_results = portfolio_developement      
        self.markets = list(self.testing_PV_results.keys())        
        self.investment_sum = self.testing_PV_results[self.markets[0]][0]
        self.setting = settings 
        self.Dates = Dates   
        self.all_data = data                    

        self.PATH = PATH
        if not os.path.exists(PATH + "\\Figures"):
            os.makedirs(PATH + "\\Figures")
                 
        self.plot_dest = self.PATH + "\\Figures" 
        self.asset_cl =  asset_Class 
        self.equaly_weighted_PF = equally_weighted_PF(self.testing_PV_results, self.investment_sum)         
        
        # führt alle Plot operationen aus         
        self.performPlotting_action()   

        plt.ioff() 


    def plot_portfolio_values(self, portfolio_values, data):
        keys = list(portfolio_values.keys())
        num_keys = len(keys) + 1

        num_subplots = min(num_keys, 6)        

        num_columns = 2

        # Calculate the number of chunks
        num_chunks = num_keys // (num_subplots * 2)
        if num_chunks == 0:
            num_chunks += 1 
        else: 
            if num_keys % (num_chunks * (num_subplots * 2)) != 0:    
                num_chunks += 1                                                      

        for chunk_idx in range(num_chunks):
            # Calculate the range of rows for this chunk
            start_row = chunk_idx * num_subplots * 2
            if num_chunks > 1 and chunk_idx != (num_chunks-1):
                end_row = start_row + num_subplots * 2
            elif chunk_idx == (num_chunks-1):
                end_row = len(keys)    
                             
            # Create the figure and gridspec
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle('Portfolio Value over Time. Asset class: {}'.format(self.asset_cl), fontsize=8, fontweight='bold')
            axes_height = 6
    
            if chunk_idx == (num_chunks-1):
                remaining_keys = end_row - start_row
                num_subplots = math.ceil(remaining_keys / 2)              
                        
            gs = fig.add_gridspec(num_subplots, num_columns)

            # Create subplots
            subplot_positions = [(i, j) for i in range(0,num_subplots) for j in range(num_columns)]
            subplots = [fig.add_subplot(gs[i, j]) for i, j in subplot_positions]

            def mio_formatter(x, pos):
                return f'{x / 1e6:.2f} Mio'
         
            # Plot data in subplots
            for i in range(start_row, end_row):
                try:                
                    key = keys[i]
                except IndexError:  
                    break                    
                    
                subplot_index = i-(start_row)
                subplot = subplots[subplot_index]

                subplot.plot(data[key].index[1:len(portfolio_values[key])],  portfolio_values[key], label=key)
                #subplot.set_ylabel('Portfolio Value', fontsize= axes_height+2)
                subplot.set_title(f'Portfolio Value ({data[key]["Name"][1]})', fontsize= axes_height+2)

                formatter = ticker.FuncFormatter(mio_formatter)
                subplot.yaxis.set_major_formatter(formatter)

                subplot.xaxis.set_tick_params(labelsize=axes_height) 
                subplot.yaxis.set_tick_params(labelsize=axes_height)   
        
                if self.asset_cl != 'Commodities':                                            
                    subplot.legend()

            if chunk_idx == (num_chunks-1):
                equally_weighted_portfolio = self.equaly_weighted_PF
                subplot = subplots[-1]  # Use the last subplot
                subplot.plot(data[key].index[1:len(equally_weighted_portfolio) + 1], equally_weighted_portfolio, label='Equally Weighted Portfolio', color='orange') 
                subplot.set_title(f'Equally Weighted Portfolio', fontsize= axes_height+2)                 
                formatter = ticker.FuncFormatter(mio_formatter)
                subplot.yaxis.set_major_formatter(formatter) 
                subplot.xaxis.set_tick_params(labelsize=axes_height) 
                subplot.yaxis.set_tick_params(labelsize=axes_height)                                      
            
            fig.tight_layout()                
            plt.savefig(str(self.plot_dest + "/Performance_" + self.asset_cl + f"_part_{chunk_idx+1}.pdf"), format='pdf')                         
            plt.close(fig)
               
 
    # Works fine gives all indvidual stock comparisons        
    # def plot_portfolio_values(self, portfolio_values, data):                                    
    #     keys = list(portfolio_values.keys())
    #     num_keys = len(keys)

    #     num_columns = 2
    #     num_rows = (num_keys + 1) // num_columns

    #     if num_rows > 10: 
    #         num_rows = 10                   

    #     # Create the figure and gridspec
    #     fig = plt.figure(figsize=(5, 5))
    #     if self.asset_cl != 'Commodities':                                            
    #         fig.suptitle('Portfolio Value over Time. Asset class: {}'.format(self.asset_cl), fontsize=12, fontweight='bold')
    #         axes_height = 8             
    #     else:              
    #         #fig.suptitle('Portfolio Value over Time. Asset class: {}'.format(self.asset_cl), fontsize=10, fontweight='bold')
    #         axes_height = 4           
            
    #     gs = fig.add_gridspec(num_rows, num_columns)

    #     # Create subplots
    #     subplot_positions = [(i, j) for i in range(num_rows) for j in range(num_columns)]
    #     subplots = [fig.add_subplot(gs[i, j]) for i, j in subplot_positions]

    #     def mio_formatter(x, pos):
    #         return f'{x / 1e6:.2f} Mio'    
        
    #     # Plot data in subplots
    #     for i, (key, values) in enumerate(portfolio_values.items()):
    #         row, col = divmod(i, num_columns)
    #         subplot_index = row * num_columns + col
    #         subplot = subplots[subplot_index]

    #         subplot.plot(data[key].index[1:], values, label=key)

    #         subplot.set_xlabel('Time Step', fontsize= axes_height+2)
    #         subplot.set_ylabel('Portfolio Value', fontsize= axes_height+2)
    #         subplot.set_title(f'Portfolio Value ({data[key]["Name"][1]})', fontsize= axes_height+2)

    #         formatter = ticker.FuncFormatter(mio_formatter)
    #         subplot.yaxis.set_major_formatter(formatter)

    #         subplot.xaxis.set_tick_params(labelsize=axes_height) 
    #         subplot.yaxis.set_tick_params(labelsize=axes_height)   
        
    #         if self.asset_cl != 'Commodities':                                            
    #             subplot.legend()

    #     equally_weighted_portfolio = self.equaly_weighted_PF
    #     subplot = subplots[-1]  # Use the last subplot
    #     subplot.plot(data[key].index[1:len(equally_weighted_portfolio)+1], equally_weighted_portfolio, label='Equally Weighted Portfolio', color='orange')  
    #     subplot.set_ylabel('Portfolio Value', fontsize=axes_height+2)
    #     subplot.set_title(f'Equally weighted Portfolio', fontsize=axes_height+2)
    #     formatter = ticker.FuncFormatter(mio_formatter)
    #     subplot.yaxis.set_major_formatter(formatter)
    #     subplot.xaxis.set_tick_params(labelsize=axes_height) 
    #     subplot.yaxis.set_tick_params(labelsize=axes_height)                                                   

    #     plt.tight_layout()
    #     plt.savefig(str(self.plot_dest + "/Performance_"+ self.asset_cl + ".pdf"), format='pdf') 

    # == Comparison between DQN and DDQN
    def plot_portfolio_performance_comparison(self):
        abs_path = self.plot_dest
        
        for key in self.data.keys():                               
            plot_title = "ALGO_performance" + key + ".pdf"
            prices = self.data[key]['ClosePrice']
  
            def getData(Type):
                log_file_path = os.path.dirname(os.path.dirname(abs_path)) + "\\results_" + Type + "_" + os.path.basename(os.path.dirname(abs_path)).split('_')[2]  + "\\" + Type + "_test" + key + ".log"

                Buy_decisions = parseLogFile(log_file_path, "Buy")
                Sell_decisions = parseLogFile(log_file_path, "Sell")

                BUY_dates = [step['Date'] for step in Buy_decisions]
                SELL_dates= [step['Date'] for step in Sell_decisions]

                sell_prices = prices[prices.index.isin(SELL_dates)]
                buy_prices = prices[prices.index.isin(BUY_dates)]

                # Cummulative performance
                portfolio_values = parse_log_file(log_file_path, 'Portfolio value:\\s*(\\d+\\.\\d+)')
                cummulated_pos = parse_log_file(log_file_path,'Current number of Positions:\\s*(-?\\d+)')
                max_position = parse_log_file(log_file_path, 'Current number of Positions:\\s*-?\\d+\\s*\\|\\s*(\\d+)')                
                return portfolio_values, cummulated_pos, max_position, BUY_dates, buy_prices, SELL_dates, sell_prices

            portfolio_values, cummulated_pos,  max_position, BUY_dates, buy_prices, SELL_dates, sell_prices = getData("DQN")
            portfolio_values1, cummulated_pos1, max_position1, BUY_dates1, buy_prices1, SELL_dates1, sell_prices1 = getData("DDQN")

            fig = plt.figure(figsize=(10, 10))
            # Create the gridspec
            gs = fig.add_gridspec(3, 2)

            # Define the positions of the subplots
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 0])
            ax5 = fig.add_subplot(gs[2, 1])

            def mio_formatter(x, pos):
                return f'{x / 1e6:.2f} Mio'                

            # Plot on the big plot
            x = np.linspace(0, 10, 100)
            ax1.plot(prices.index[0:len(portfolio_values)], portfolio_values, color='steelblue', label = "DQN") 
            formatter = ticker.FuncFormatter(mio_formatter)
            ax1.yaxis.set_major_formatter(formatter)            
            ax1.plot(prices.index[0:len(portfolio_values)], portfolio_values1, color='lightgreen', label ="Double DQN") # DDQN
            ax1.axhline(y=self.investment_sum, color='tomato', linestyle='--', linewidth=0.75)
            ax1.set_title('DQN-performance vs. DDQN ({}-{})'.format(self.setting[1], self.setting[2]), fontsize=14, weight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.tick_params(axis='x', labelsize=8)
            ax1.legend()
            ax1.grid()

            #DQN performance
            ax2.plot(prices.index, prices.tolist(), color='black', label= key)
            ax2.scatter(pd.DatetimeIndex(BUY_dates), buy_prices.tolist(), c='green', alpha=0.5, label='buy')
            ax2.scatter(pd.DatetimeIndex(SELL_dates), sell_prices.tolist(), c='red', alpha=0.5, label='sell')
            ax2.set_title('DQN Actions on {}'.format(key), fontsize=14, weight='bold')
            ax2.set_ylabel('Price')
            ax2.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='x', labelsize=8)
            ax2.legend()
            ax2.grid()
            ax2.set_title('Actions taken by DQN')

            #DDQN performance
            ax3.plot(prices.index, prices.tolist(), color='black', label= key)
            ax3.scatter(pd.DatetimeIndex(BUY_dates1), buy_prices1.tolist(), c='green', alpha=0.5, label='buy')
            ax3.scatter(pd.DatetimeIndex(SELL_dates1), sell_prices1.tolist(), c='red', alpha=0.5, label='sell')
            ax3.set_title('DQN Actions on {}'.format(key), fontsize=14, weight='bold')
            ax3.set_ylabel('Price')
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='x', labelsize=8)
            ax3.legend()
            ax3.grid()
            ax3.set_title('Actions taken by DDQN')

            # DQN asset allocation
            ax4.plot(prices.index[0:len(cummulated_pos)], cummulated_pos, color='blue')
            ax4.plot(prices.index[0:len(max_position)], max_position, color='tomato', linestyle='--', linewidth=0.75)
            ax4.plot(prices.index[0:len(max_position)], [-value for value in max_position], color='tomato', linestyle='--', linewidth=0.75)                       
            ax4.set_title('Change in position size', fontsize=14, weight='bold')
            ax4.set_ylabel('Position size')
            ax4.tick_params(axis='x', rotation=45)
            ax4.tick_params(axis='x', labelsize=8)
            ax4.set_title('Change in position size (DQN)')

            # DDQN asset allocation
            ax5.plot(prices.index[0:len(cummulated_pos1)], cummulated_pos1, color='blue')
            ax5.plot(prices.index[0:len(max_position1)], max_position1, color='tomato', linestyle='--', linewidth=0.75)   
            ax5.plot(prices.index[0:len(max_position1)], [-value for value in max_position1], color='tomato', linestyle='--', linewidth=0.75)                      
            ax5.set_title('Change in position size', fontsize=14, weight='bold')
            ax5.set_ylabel('Position size')
            ax5.tick_params(axis='x', rotation=45)
            ax5.tick_params(axis='x', labelsize=8)
            ax5.set_title('Change in position size (DDQN)')

            plt.tight_layout()
            plt.tight_layout()  # Adjust the spacing between subplots if needed
            plt.savefig(str(abs_path + "\\" + plot_title), format='pdf')
            plt.close()   
                                                       
        
    def TrainTestPeriods(self):
        abs_path = self.plot_dest
        key = random.choice(list(self.all_data.keys()))        

        for i in range(0, len(self.Dates)):
            Name = "Train_Test_Setting" + str(i+1)              
            d = self.all_data[key]['ClosePrice']
            d = d.loc[(d.index > np.datetime64(self.Dates[i][0])) & (d.index < np.datetime64(self.Dates[i][1]))]

            c = self.all_data[key]['ClosePrice']
            c = c.loc[(c.index > np.datetime64(self.Dates[i][1])) & (c.index < np.datetime64(self.Dates[i][2]))]        
            fig = plt.figure(figsize=(8, 6))

            # Plot series d
            plt.plot(d.index, d, color='steelblue', label="Training phase")

            # Plot series c starting where series d ends
            plt.plot(c.index, c, color='tomato', label="Testing phase")

            max_value = pd.concat([c, d]).max() * 1.1
            min_value = pd.concat([c, d]).min() * 0.9

            # Set y-axis limits
            plt.ylim(min_value, max_value)

            # Add plot title
            plt.title("Setting {} \nTraining ({} until {});\n Testing ({} until {})".format(
                i+1,
                self.Dates[i][0].strftime('%Y-%m-%d'),
                self.Dates[i][1].strftime('%Y-%m-%d'),
                self.Dates[i][1].strftime('%Y-%m-%d'),
                self.Dates[i][2].strftime('%Y-%m-%d')
            ), fontsize=14, weight='bold')
            plt.savefig(str(abs_path  + "\\" +  Name + ".pdf"), format='pdf')
            plt.close()                  


    def abs_ret(self, data, ticker, curr_time, end_date1, *args):
        abs_path = "outputs/" + curr_time + "/results_TestingPhase_" + str(end_date1.year) + "/"
        plot_title = "portfolio_performance_comparison.pdf"

        cfg, env, cfg1, env1 = args
        Q_learning_profits = getPerformance(ticker, cfg, env, pattern=None)    
        DoubleQ_learning_profits = getPerformance(ticker, cfg1, env1, pattern=None)  
        MACD_profits = getPerformance(ticker, cfg1, env1, pattern=2)
        buy_and_hold_rewards = [(data[stock]['portfolio_value'][-1] - data[stock]['portfolio_value'][0]) for stock in ticker]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))  # plot the test result
        width = 0.2
        x = np.arange(len(ticker))
        ax.set_title(str("Comparison between DQN, DDQN vs. Buy & Hold and MACD strategy's profits"), fontsize=14, weight='bold')
        ax.bar(x, Q_learning_profits, width=width, color='tomato', label='DQN')
        ax.bar(x+width, buy_and_hold_rewards, width=width, color='steelblue', label='Buy and Hold')
        ax.bar(x+(2*width), MACD_profits, width=width, color='firebrick', label='MACD strategy')
        ax.bar(x+(3*width), DoubleQ_learning_profits, width=width, color='#B2DFEE', label='Double DQN')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.75)
        ax.set_xticks(x+width/2)
        ax.set_xticklabels(data, fontsize=12, rotation=90)
        ax.legend()
        plt.savefig(str(abs_path + plot_title), format='pdf')
        plt.close()

           
    # == Führt das Plotting aus ========= 
    def performPlotting_action(self):
        #self.plot_portfolio_values(self.testing_PV_results, self.data)     
        #self.abs_ret()   
        self.TrainTestPeriods()  
        #self.plot_portfolio_performance_comparison()      
    


    # def plot_portfolio_performance_comparison(self, key, reward_data, end_date1, end_date2):
    #     cum_ret = self.test_data[key][self.state_space_dim:]
    #     investment_value_BH = (cum_ret['return'] + 1).cumprod() * self.investment_sum

    #     # Cummulative performance
    #     DQN_ret = reward_data[key]
    #     DQN_ret = DQN_ret.apply(lambda x: x['Return'])
    #     investment_value_DQN = (DQN_ret + 1).cumprod() * self.investment_sum

    #     # Buy and Sell signals
    #     prices = self.getData([key], end_date1, end_date2)
    #     prices = pd.DataFrame(prices[key])
    #     prices = prices[self.state_space_dim:]
    #     reward_data = reward_data[key]

    #     def getEntries(type):
    #         mask = reward_data.apply(lambda x: x[type] == 1)
    #         result = reward_data.index[mask]
    #         return result

    #     SELL_dates = getEntries('SELL')
    #     BUY_dates = getEntries('BUY')

    #     sell_prices = prices[prices.index.isin(SELL_dates)]
    #     buy_prices = prices[prices.index.isin(BUY_dates)]

    #     # Plotting
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100)

    #     ax1.plot(investment_value_BH, color='lightgreen', label= 'Long-only')
    #     ax1.plot(investment_value_DQN, color='steelblue', label = 'DQN')
    #     ax1.axhline(y=10000, color='tomato', linestyle='--', linewidth=0.75)
    #     ax1.set_title('DQN-strategy vs. Buy and Hold-Benchmark ({})'.format(key), fontsize=14, weight='bold')
    #     ax1.set_ylabel('Portfolio Value ($)')
    #     ax1.legend()
    #     ax1.grid()

    #     ax2.plot(prices, color='black', label= key)
    #     ax2.scatter(BUY_dates, buy_prices['ClosePrice'], c='green', alpha=0.5, label='buy')
    #     ax2.scatter(SELL_dates, sell_prices['ClosePrice'], c='red', alpha=0.5, label='sell')
    #     ax2.set_title('DQN Actions on {}'.format(key), fontsize=14, weight='bold')
    #     ax2.set_ylabel('Price')
    #     ax2.legend()
    #     ax2.grid()

    #     plt.savefig('outputs/TradingSystem_v0/{}.pdf'.format(key))
