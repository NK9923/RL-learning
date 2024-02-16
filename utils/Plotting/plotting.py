from cProfile import label
import altair as alt
import pandas as pd
import os
import random
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

from utils.Neural_Networks.dqn import DQN


def max_drawdown_absolute(returns):
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    start = r.loc[:end].argmax()
    return mdd, start, end


def visualize(cfg, history, title="trading session", matchDate=None):
    if matchDate is None:
        return        
    
    df = cfg.test_data    
    df = df[['Adjusted_Close_EUR']]    
    df.reset_index(inplace=True)    
    df = df.rename(columns={'Adjusted_Close_EUR': 'actual', 'Date' : 'date'})  
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df= df[df['date'] >= matchDate]
                  
    # add history to dataframe
    position = [history[0][1]] + [x[1] for x in history]
    actions = ['HOLD'] + [x[2] for x in history]
    timestamp_to_value = {timestamp: action for timestamp, value, action, _ in history}    
    #df['position'] = df['date'].map(timestamp_to_value)
    df['action'] = df['date'].map(timestamp_to_value)
    df['action'].fillna('HOLD', inplace=True)    
    
    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(df['actual'])), max(max(df['actual']), max(df['actual']))), clamp=True)
    
    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('actual', axis=alt.Axis(format='$.2f', title='Price'), scale=scale)
    ).interactive(
        bind_y=False
    )
    
    # plot the BUY and SELL actions as points
    points = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('actual', axis=alt.Axis(format='$.2f', title='Price'), scale=scale),
        color='action'
    ).interactive(bind_y=False)

    # merge the two charts
    chart = alt.layer(actual, points, title=title).properties(height=300, width=1000)

    chart.save("chart.html", embed_options={'renderer': 'svg'}) 


def abs_ret(data, asset_Class, dir_name):
    abs_path = "outputs\\" + dir_name 
    plot_title = "AbsoluteGain" + asset_Class + ".pdf"

    performances = []

    for setting in range(0,4):
        Setting = data[setting]        
        Risk_factors, Risk_factors_BH, Risk_factors_MACD = Setting
        Performance_DQN = Risk_factors['DQN']['Portf']['abs_ret'] * 100
        Performance_DDQN = Risk_factors['DDQN']['Portf']['abs_ret'] *100   
        Performance_BH = Risk_factors_BH['abs_ret'] * 100  
        Performance_MACD = Risk_factors_MACD['abs_ret'] * 100
    
        performance_setting = [Performance_MACD, Performance_BH, Performance_DQN, Performance_DDQN]
        performances.append(performance_setting)     

    performances = np.array(performances)   

    colors = ['lightsteelblue', 'lightblue', 'lightskyblue', 'steelblue']       

    data_std = np.array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], 
                     [1, 2, 1, 2]])    

    length = len(performances)
    x_labels = ['2017', '2018', '2019', '2020']

    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(length)

    ax.bar(x, performances[:,0], width, color=colors[0], label='MACD', yerr=data_std[:,0])
    ax.bar(x + width, performances[:,1], width, color=colors[1], label='Buy and Hold', yerr=data_std[:,1])
    ax.bar(x + (2 * width), performances[:,2], width, color=colors[2], label='DQN', yerr=data_std[:,2])
    ax.bar(x + (3 * width), performances[:,3], width, color=colors[3], label='DDQN', yerr=data_std[:,3])

    ax.set_ylabel('Performance in %')
    ax.set_ylim(np.min(performances)-1,np.max(performances)+1)
    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Scenario')   
    ax.set_title('Performance Comparison: ' + asset_Class)
    ax.legend()
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    fig.tight_layout()

    plt.savefig(os.path.join(os.path.join(os.getcwd(), abs_path), plot_title), format='pdf')
    plt.close()      


class Plotting_1:
    def __init__(self, cfg, PATH, data, asset_Class, settings, Transaction_History, individual):       
        self.data = data
        self.data.columns = self.data.columns.str.replace('_Portfolio_Value', '')         
        self.Config = cfg        
        self.equaly_weighted_PF = data['Portfolio_Value']      
        self.markets = cfg.asset_class_Ticker       
        self.investment_sum = self.equaly_weighted_PF[0]
        self.setting = settings
        self.Dates = self.Config.config['Dates']   
        cfg.getTrainAndTestData()     
        self.all_data = cfg.data  
        self.TH = Transaction_History 
        self.individualPF = individual                 

        self.PATH = PATH
        if not os.path.exists(PATH + "\\Figures"):
            os.makedirs(PATH + "\\Figures")
                 
        self.plot_dest = self.PATH + "\\Figures" 
        self.asset_cl =  asset_Class 
            
        self.performPlotting_action()   

        plt.ioff() 
    
    def plot_portfolio_values(self):
        plt.ioff()

        def mio_formatter(x, pos):
            return f'{x / 1e6:.2f} Mio'        
        
        keys = self.Config.asset_class_Ticker  
        num_keys = len(keys) + 1

        num_subplots = min(num_keys, 7)        
        num_columns = 2

        num_chunks = num_keys // (num_subplots * 2)
        if num_chunks == 0:
            num_chunks += 1 
        else: 
            if num_keys % (num_chunks * (num_subplots * 2)) != 0:    
                num_chunks += 1                                                      

        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * num_subplots * 2
            if num_chunks > 1 and chunk_idx != (num_chunks-1):
                end_row = start_row + num_subplots * 2
            elif chunk_idx == (num_chunks-1):
                end_row = len(keys)    
                             
            # Create the figure and gridspec
            fig = plt.figure(figsize=(10, 12))
            fig.suptitle('Portfolio Value over Time. Asset class: {}'.format(self.asset_cl), fontsize=10, fontweight='bold')
            axes_height = 5

            
            delete_subplot = False    
            if chunk_idx == (num_chunks-1):
                remaining_keys = end_row - start_row
                num_subplots = math.ceil(remaining_keys / 2) 
                if remaining_keys % 2 == 0:
                    num_subplots += 1  
                    delete_subplot = True                                                                   
                        
            gs = fig.add_gridspec(num_subplots, num_columns)

            # Create subplots
            subplot_positions = [(i, j) for i in range(0,num_subplots) for j in range(num_columns)]
            subplots = [fig.add_subplot(gs[i, j]) for i, j in subplot_positions]

            # Plot data in subplots
            for i in range(start_row, end_row):
                try:                
                    key = keys[i]
                except IndexError:  
                    break                    
                    
                subplot_index = i-(start_row)
                subplot = subplots[subplot_index]

                subplot.plot(self.data['Date'][0:len(self.data[key])],  self.data[key], label=key)
                #subplot.set_ylabel('Portfolio Value', fontsize= axes_height+2)
                subplot.set_title(f'Portfolio Value ({self.all_data[key]["Name"][1]})', fontsize= axes_height+2)

                formatter = ticker.FuncFormatter(mio_formatter)
                subplot.yaxis.set_major_formatter(formatter)

                subplot.xaxis.set_tick_params(labelsize=axes_height) 
                subplot.yaxis.set_tick_params(labelsize=axes_height+1)   
        
                if self.asset_cl != 'Commodities':                                            
                    subplot.legend()

            if chunk_idx == (num_chunks-1):
                
                if delete_subplot:
                    subplot = subplots[-2]
                else:                                    
                    subplot = subplots[-1]  # Use the last subplot
                    
                subplot.plot(self.data['Date'][0:len(self.data[key])], self.data['Portfolio_Value'], label='Equally Weighted Portfolio', color='orange') 
                subplot.set_title(f'Equally Weighted Portfolio', fontsize= axes_height+2)                 
                formatter = ticker.FuncFormatter(mio_formatter)
                subplot.yaxis.set_major_formatter(formatter) 
                subplot.xaxis.set_tick_params(labelsize=axes_height) 
                subplot.yaxis.set_tick_params(labelsize=axes_height)        

            if delete_subplot:
                subplot = subplots[-1]    
                subplot.axis('off')                                                                                        
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])                
            plt.savefig(str(self.plot_dest + "/Performance_" + self.asset_cl + "_" + self.Config.TestinDate[2][:4] + "_" + self.Config.algo_name  + f"_part_{chunk_idx+1}.pdf") , format='pdf', orientation='portrait')
            plt.close(fig)                          
        
    def TrainTestPeriods(self):
        plt.ioff()        
        abs_path = self.plot_dest
        key = random.choice(self.markets)        

        for i in range(0, len(self.Dates)):
            Name = "Train_Test_Setting" + str(i+1)

            fig, ax = plt.subplots(figsize=(8, 6))  # Create a single figure and axes

            # Plot your data
            d = self.all_data[key]['AdjustedClose']
            d = d.loc[(d.index > np.datetime64(self.Dates[i][0])) & (d.index < np.datetime64(self.Dates[i][1]))]

            c = self.all_data[key]['AdjustedClose']
            c = c.loc[(c.index > np.datetime64(self.Dates[i][1])) & (c.index < np.datetime64(self.Dates[i][2]))]

            b = self.all_data[key]['Close']
            b = b.loc[(b.index > np.datetime64(self.Dates[i][0])) & (b.index < np.datetime64(self.Dates[i][2]))]

            ax.plot(d.index, d, color='steelblue', label="Training phase (back-adjusted Close)")
            ax.plot(c.index, c, color='tomato', label="Testing phase (back-adjusted Close)")
            ax.plot(b.index, b, color='grey', alpha=0.3, label="Close price")

            ax.legend(fontsize=10)
            max_value = pd.concat([b, c, d]).max() * 1.1
            min_value = pd.concat([b, c, d]).min() * 0.9
            ax.set_ylim(min_value, max_value)

            # Create separate text elements for different parts of the title
            title_main = r"$\bf{Setting\ " + str(i+1) + "}$"
            title_sub = r"$\bf{{{} ({})}}$".format(self.all_data[key]['Name'][0], key)
            title_training = "Training ({} until {})".format(self.Dates[i][0], self.Dates[i][1])
            title_testing = "Testing ({} until {})".format(self.Dates[i][1], self.Dates[i][2])

            # Define font sizes for each part
            fontsize_main = 14
            fontsize_training = 10
            fontsize_testing = 10

            # Add the text elements to the existing axes with specified positions and font sizes
            ax.text(0.5, 1.2, title_main, fontsize=fontsize_main, ha='center', transform=ax.transAxes)
            ax.text(0.5, 1.15, title_sub, fontsize=fontsize_main, ha='center', transform=ax.transAxes)
            ax.text(0.5, 1.1, title_training, fontsize=fontsize_training, ha='center', transform=ax.transAxes)
            ax.text(0.5, 1.05, title_testing, fontsize=fontsize_testing, ha='center', transform=ax.transAxes)

            plt.tight_layout()
            fig_name = Name + ".pdf"
                                    
            plt.savefig(os.path.join(abs_path, fig_name), format='pdf')
            plt.close()                  
            
    # == Führt das Plotting aus ========= 
    def performPlotting_action(self):
        plt.ioff()             
        self.plot_portfolio_values()     
        self.TrainTestPeriods()         
        
   
class Plotting_2:
    def __init__(self, cfg, data, asset_Class, settings, 
                 Transaction_History, individual, 
                 MACD, Risk_MACD, 
                 BH, Risk_BH, Risk_RL):     
          
        # -- this represents the portfolio perfromance of both DQN and DDQN
        self.data = data
        self.data_DQN = self.data['DQN']    # Porftolio performance DQN
        self.data_DDQN = self.data['DDQN']  # Porftolio performance DDQN  
        self.data_DQN.columns = self.data_DQN.columns.str.replace('_Portfolio_Value', '')
        self.data_DDQN.columns = self.data_DDQN.columns.str.replace('_Portfolio_Value', '')                  
           
        # -- the portfolio value is the thing needed here    
        self.Config = cfg        
        self.equaly_weighted_PF_DQN = data['DQN']['Portfolio_Value'] 
        self.equaly_weighted_PF_DDQN = data['DDQN']['Portfolio_Value']

        # -- individual risk factors of both DQN and DDQN
        self.Risk_RL_strategies = Risk_RL  
        self.Risk_RL_DQN = self.Risk_RL_strategies['DQN']
        self.Risk_RL_DDQN = self.Risk_RL_strategies['DDQN']                      
            
        # -- MACD section
        self.MACD_portfolio = MACD    
        self.MACD_Risk = Risk_MACD   

        # -- for Buy and Hold strategy
        self.BH_portfolio = BH
        self.BH_Risk = Risk_BH                 

        # -- setting     
        self.markets = cfg.asset_class_Ticker       
        self.investment_sum = self.equaly_weighted_PF_DQN[0]
        self.setting = settings
        self.Dates = self.Config.config['Dates']   
        cfg.getTrainAndTestData()   

        # -- underlyings          
        self.all_data = cfg.data 
        self.test_data = cfg.test_data         

        # -- more detailed information about the algorithm's performance        
        self.TH_DQN = Transaction_History['DQN']                       # actions taken by the algorithms 
        self.TH_DDQN = Transaction_History['DDQN']   
            
        self.individualPF_DQN = individual['DQN']                      # individual porfolios for each market
        self.individualPF_DDQN = individual['DDQN']                             

        self.plot_dest = []
        self.PATH = cfg.PATHS
        for path in self.PATH:
            if not os.path.exists(os.path.join(path, "Figures")):
                os.makedirs(os.path.join(path, "Figures"))
            self.plot_dest.append(os.path.join(path, "Figures"))                
                 
        self.asset_cl =  asset_Class             
        self.performPlotting_action()   

    def boxplot(self):   
        plot_title = "Sharpe_ratio_" + str(self.setting) + "_" + self.asset_cl +".pdf"
        
        sharp_arrays = [np.array(value.get('Sharp')).flatten() for key, value in self.Risk_RL_DQN.items() if key != 'Portf' and value.get('Sharp') is not None]
        big_array_DQN = np.concatenate(sharp_arrays)

        sharp_arrays = [np.array(value.get('Sharp')).flatten() for key, value in self.Risk_RL_DDQN.items() if key != 'Portf' and value.get('Sharp') is not None]
        big_array_DDQN = np.concatenate(sharp_arrays) 

        sharp_arrays = [np.array(value.get('Sharp')).flatten() for value in self.MACD_Risk.values() if value.get('Sharp') is not None]
        big_array_MACD = np.concatenate(sharp_arrays)     

        sharp_arrays = [np.array(value.get('Sharp')).flatten() for value in self.BH_Risk.values() if value.get('Sharp') is not None]
        big_array_BH = np.concatenate(sharp_arrays)                     

        all_data = [big_array_MACD, big_array_BH, big_array_DQN, big_array_DDQN]
        labels = ['MACD', 'Long-only', 'DQN', 'DDQN']

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 4)) 

        # Rectangular box plot
        bplot1 = ax1.boxplot(all_data,
                             vert=True,
                             patch_artist=True,  
                             labels=labels) 
        ax1.set_title('Comparison of Sharpe Ratios: ' + self.asset_cl)

        # Define colors
        colors = ['lightsteelblue', 'lightblue', 'lightskyblue', 'steelblue']

        plt.gca().set_frame_on(False)

        # Apply colors to the boxes
        for patch, color in zip(bplot1['boxes'], colors):
            patch.set_facecolor(color)

        # Set the color of the median line to black
        for median in bplot1['medians']:
            median.set(color='black')

        # Adding horizontal grid lines
        ax1.yaxis.grid(False)
        ax1.set_xlabel('')
        ax1.set_ylabel('Sharpe Ratio')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dest[0], plot_title), format='pdf')
        plt.savefig(os.path.join(self.plot_dest[1], plot_title), format='pdf')
            
        plt.close()  
           

    def plot_portfolio_performance_comparison(self):
        plt.ioff()        
        abs_path = self.plot_dest

        def mio_formatter(x, pos):
            return f'{x:.2f} Mio'           
        
        for key in self.Config.test_data:                               
            plot_title = "ALGO_performance" + key + ".pdf"
            prices = self.Config.test_data[key]['Adjusted_Close_EUR']

            ReferenceTS = (1 + self.Config.test_data[key]['LogRoR'][1:].cumsum()) *  self.investment_sum   
            value_to_add = pd.Series(self.investment_sum, index=[self.Config.test_data[key].index[0]])
            combined_series = pd.concat([value_to_add, ReferenceTS])

            max_draw_DQN = max_drawdown_absolute(self.individualPF_DQN[key]['RoR_PF'])  
            max_draw_DDQN = max_drawdown_absolute(self.individualPF_DDQN[key]['RoR_PF'])   
                                              
            # For DQN   
            portfolio_values_DQN = self.data_DQN[key]
            cummulated_pos_DQN = self.TH_DQN[key]['Positions'] 
            BUY_dates_DQN =  self.TH_DQN[key][self.TH_DQN[key]['Type'] == 2]['Date']  
            buy_prices_DQN =  self.TH_DQN[key][self.TH_DQN[key]['Type'] == 2]['Value']             
            SELL_dates_DQN =  self.TH_DQN[key][self.TH_DQN[key]['Type'] == 0]['Date']   
            sell_prices_DQN =  self.TH_DQN[key][self.TH_DQN[key]['Type'] == 0]['Value']    

            # For DDQN   

            portfolio_values_DDQN = self.data_DDQN[key]
            cummulated_pos_DDQN = self.TH_DDQN[key]['Positions'] 
            BUY_dates_DDQN =  self.TH_DDQN[key][self.TH_DDQN[key]['Type'] == 2]['Date']  
            buy_prices_DDQN =  self.TH_DDQN[key][self.TH_DDQN[key]['Type'] == 2]['Value']             
            SELL_dates_DDQN =  self.TH_DDQN[key][self.TH_DDQN[key]['Type'] == 0]['Date']   
            sell_prices_DDQN =  self.TH_DDQN[key][self.TH_DDQN[key]['Type'] == 0]['Value']  
                                       
            fig = plt.figure(figsize=(10, 12))
            gs = fig.add_gridspec(4, 2)

            # Define the positions of the subplots
            ax1 = fig.add_subplot(gs[0, :])
            ax6 = fig.add_subplot(gs[1, :])            
            ax2 = fig.add_subplot(gs[2, 0])
            ax3 = fig.add_subplot(gs[2, 1])
            ax4 = fig.add_subplot(gs[3, 0])
            ax5 = fig.add_subplot(gs[3, 1])    

            value_to_prepend = 1.000e+08
            num_rows_to_prepend = len(self.test_data[key]) - len(portfolio_values_DQN)
            new_list = [value_to_prepend] * num_rows_to_prepend
            
            portfolio_values_DQN = portfolio_values_DQN.tolist()
            portfolio_values_DQN = new_list + portfolio_values_DQN    
         
            # Plot on the big plot
            x = np.linspace(0, 10, 100)
            ax1.plot(prices.index[0:len(portfolio_values_DQN)], ([x / 1e6 for x in portfolio_values_DQN]), color='steelblue', label = "DQN")    
            ax1.axvspan(prices.index[max_draw_DQN[1] + num_rows_to_prepend], prices.index[max_draw_DQN[2] + num_rows_to_prepend-1], alpha=0.1, facecolor="r", edgecolor="r", hatch='/') 
            ax1.axvspan(prices.index[max_draw_DDQN[1] + num_rows_to_prepend], prices.index[max_draw_DDQN[2] + num_rows_to_prepend -1], alpha=0.15, facecolor="darkred", edgecolor="darkred", hatch= '\\')         

  
            value_to_prepend = 1.000e+08
            num_rows_to_prepend = len(self.test_data[key]) - len(portfolio_values_DDQN)
            new_list = [value_to_prepend] * num_rows_to_prepend
            
            portfolio_values_DDQN = portfolio_values_DDQN.tolist()
            portfolio_values_DDQN = new_list + portfolio_values_DDQN                                
    

            formatter = ticker.FuncFormatter(mio_formatter)
            ax1.yaxis.set_major_formatter(formatter) 
            ax1.plot(prices.index[0:len(portfolio_values_DDQN)], ([x / 1e6 for x in portfolio_values_DDQN]), color='lightgreen', label ="Double DQN") # DDQN
            ax1.axhline(y=(self.investment_sum / 1e6), color='tomato', linestyle='--', linewidth=0.75)
            ax1.set_title('DQN-performance vs. DDQN\n{}'.format(self.Config.test_data[key]["Name"][0]), fontsize=14, weight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.tick_params(axis='x', labelsize=8)
            ax1.legend()
            ax1.grid()

            # Benchmark portfolio
            ax6.plot(prices.index[0:len(portfolio_values_DQN)], (combined_series[0:len(portfolio_values_DQN)] / 1e6), color='grey', label = "Reference-PF Long only")
            ax6.yaxis.set_major_formatter(formatter) 
            ax6.axhline(y=(self.investment_sum / 1e6), color='tomato', linestyle='--', linewidth=0.75)   
            ax6.set_title('Passiv Long-only benchmark', fontsize=14, weight='bold')                
            ax6.set_ylabel('Portfolio Value ($)')
            ax6.tick_params(axis='x', rotation=45)
            ax6.tick_params(axis='x', labelsize=8)
            ax6.legend()
            ax6.grid()                 
            
            #DQN performance
            ax2.plot(prices.index, prices.tolist(), color='black', label= key)
            ax2.scatter(pd.DatetimeIndex(BUY_dates_DQN), buy_prices_DQN.tolist(), c='green', alpha=0.5, label='buy')
            ax2.scatter(pd.DatetimeIndex(SELL_dates_DQN), sell_prices_DQN.tolist(), c='red', alpha=0.5, label='sell')
            ax2.set_title('DQN Actions on {}'.format(key), fontsize=14, weight='bold')
            ax2.set_ylabel('Price')
            ax2.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='x', labelsize=8)
            ax2.legend()
            ax2.grid()
            ax2.set_title('Actions taken by DQN')

            #DDQN performance
            ax3.plot(prices.index, prices.tolist(), color='black', label= key)
            ax3.scatter(pd.DatetimeIndex(BUY_dates_DDQN), buy_prices_DDQN.tolist(), c='green', alpha=0.5, label='buy')
            ax3.scatter(pd.DatetimeIndex(SELL_dates_DDQN), sell_prices_DDQN.tolist(), c='red', alpha=0.5, label='sell')
            ax3.set_title('DDQN Actions on {}'.format(key), fontsize=14, weight='bold')
            ax3.set_ylabel('Price')
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='x', labelsize=8)
            ax3.legend()
            ax3.grid()
            ax3.set_title('Actions taken by DDQN')

            
            value_to_prepend = 0
            num_rows_to_prepend = len(self.test_data[key]) - len(cummulated_pos_DQN)
            new_list = [value_to_prepend] * num_rows_to_prepend           
            cummulated_pos_DQN = new_list + cummulated_pos_DQN.tolist()     
            

            # DQN asset allocation
            ax4.plot(prices.index[0:len(cummulated_pos_DQN)], cummulated_pos_DQN, color='blue')                 
            ax4.set_title('Change in position size', fontsize=14, weight='bold')
            ax4.set_ylabel('Position size')
            ax4.tick_params(axis='x', rotation=45)
            ax4.tick_params(axis='x', labelsize=8)
            ax4.set_title('Change in position size (DQN)')

            value_to_prepend = 0
            num_rows_to_prepend = len(self.test_data[key]) - len(cummulated_pos_DDQN)
            new_list = [value_to_prepend] * num_rows_to_prepend
            cummulated_pos_DDQN = new_list + cummulated_pos_DDQN.tolist()    
            

            # DDQN asset allocation
            ax5.plot(prices.index[0:len(cummulated_pos_DDQN)], cummulated_pos_DDQN, color='blue')                
            ax5.set_title('Change in position size', fontsize=14, weight='bold')
            ax5.set_ylabel('Position size')
            ax5.tick_params(axis='x', rotation=45)
            ax5.tick_params(axis='x', labelsize=8)
            ax5.set_title('Change in position size (DDQN)')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dest[0], plot_title), format='pdf')
            plt.savefig(os.path.join(self.plot_dest[1], plot_title), format='pdf')
            
            plt.close()   
    
    # == Führt das Plotting aus ========= 
    def performPlotting_action(self):
        plt.ioff()        
        self.plot_portfolio_performance_comparison()         
        self.boxplot() 
    
