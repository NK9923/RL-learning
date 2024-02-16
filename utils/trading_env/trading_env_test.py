import os
import random
import numpy as np
import pandas as pd
import math
import yaml
from collections import Counter

from utils.trading_env.risk_metrics import RISK_metrics
from utils.Logging.Log import *
from utils.trading_env.ex_ante_vola import calculate_ex_ante_volatility

action_to_str = {2: 'Buy',
                 0: 'Sell',
                 1: 'Hold'}

def format_action_count(label, actions):
    action_counts = Counter(actions)
    return f"{action_counts[label]}"

class ActionInvalid(Exception):
    def __init__(self, action, message="action must be 0 or 1 or 2"):
        self.action = action
        self.message = "action : " + str(action) + " is not valid. \n" + message
        super().__init__(self.message)

class TradingEnv():
    def __init__(self, cfg, returns_data, k_value, mode, key = None, df=None, file='data/AAPL.csv', absolute_path=False, config_file_path="config/config.yml"):
        super(TradingEnv, self).__init__()
        self.ASSET = key   
        self.ticker =  self.ASSET        

        self.mode = mode  # test or train
        self.cfg = cfg        
        self.index = 0
        self.data = returns_data
        
        if self.ASSET is None and len(self.data.keys()) > 0:
            self.tickers =  self.data.keys()           
            self.current_stock = random.choice(list(self.data.keys()))
            self.ticker = self.current_stock   
            self.object_type = 'Multi_Instrument'  
            self.r_ts = self.data[self.ticker]['RoR'].tolist()                                
        else:
            self.ticker = self.ASSET
            self.current_stock = self.ticker
            self.object_type = 'Individual_Instrument'            
            self.r_ts = self.data

        self.k = k_value                            
        self.total_steps = len(self.r_ts) - self.k
        self.current_step = 0
        self.initial_state = tuple(self.r_ts[:self.k])  # Use tuple because it's immutable
        self.state = self.initial_state
        self.reward = 0.0
           
        self.is_terminal = False  
        self.profits = 0        
        
        PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))   
        self._load_config(config_file_path)  
        self.reset()                         

        if (df is None or not self.Futures) and self.mode == 'test':
            if absolute_path:
                file_path = file
            else:
                file_path = os.path.join(PACKAGE_DIR, file)
            self.df = pd.read_csv(file_path)
            self.df = self.df.sort_values('date')
            self.TransactionCostFactor = 0.025          # for stocks            
            pass            
        else:
            self.df = df

        self.reward_range = (0, self.MAX_ACCOUNT_BALANCE - self.INITIAL_ACCOUNT_BALANCE)
        
    def reset(self, indx = None) -> None:
        self.INITIAL_ACCOUNT_BALANCE = self.config.get('INITIAL_ACCOUNT_BALANCE')
        self.MAX_ACCOUNT_BALANCE =  self.config.get('MAX_ACCOUNT_BALANCE')   
        self.TIME_STEP = self.config.get('TIME_STEP')          
        self.net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.min_net_worth =  self.INITIAL_ACCOUNT_BALANCE       
        self.sizing_factor = self.config.get('sizing_factor')
        self.Futures = self.config.get('Futures')        
        
        self.contracts_held = 0
        self.cost_basis = 0
        self.balance = self.INITIAL_ACCOUNT_BALANCE       
        self.total_contracts_sold = 0
        self.total_sales_value = 0
        self.total_contracts_bought = 0  
        self.TransactionCostFactor = self.config.get('TransactionCostFactor')        # 1 bp  
        self.rf = self.config.get('RiskFreeRate')        
        self.Margin_requirement = self.config.get('Margin_requirement', 0.1)        
        self.balance = self.net_worth  
        self.MarketWeight = 0.1/3  
        self.Vola_targets = self.config.get('target_vola', 0.4)   
        self.target_vola = self.Vola_targets[self.cfg.asset_Class]            
        self.actions = []
        self.transactionCost = []  
        self.portfolio_history  = [self.INITIAL_ACCOUNT_BALANCE]   
        self.position_history = [0] 
        self.append_position_tracking = []   
        self.position_sizing = 0                                                  
        
        self.start_point = 1 if not indx else indx
        self.current_step = 0

        if self.mode == 'train':
            if self.ASSET is None and len(self.data.keys()) > 0:
                self.tickers =  self.data.keys()           
                self.current_stock = random.choice(list(self.data.keys()))
                self.ticker = self.current_stock   
                self.r_ts = self.data[self.ticker]['RoR'].tolist()                                
            else:
                self.ticker = self.ASSET
                self.current_stock = self.ticker
                self.object_type = 'Individual_Instrument'            
                self.r_ts = self.data

        else:
            if self.object_type == 'Multi_Instrument':
                self.current_stock = self.ticker[self.index]
                self.index += 1
                self.r_ts = self.data[self.current_stock]    

            else:
                pass                               
          
        self.total_steps = len(self.r_ts) - self.k  
        self.initial_state = tuple(self.r_ts[:self.k])
        self.state = self.initial_state
        self.reward = 0.0
        self.positions = []        
        self.is_terminal = False
                
        return self.state


    def _load_config(self, config_file_path) -> None:
        try:
            with open(os.path.join(os.getcwd(), config_file_path), 'r') as stream:
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{config_file_path}' not found.")    


    def _get_additional_info(self) -> pd.DataFrame:
        try:
            df = pd.read_csv('Input/Markets.csv')
        except FileNotFoundError:
            raise FileNotFoundError("CSV file 'Input/Markets.csv' not found.")   

        try:
            df = pd.DataFrame(df)  
            self.ticksize, self.contractsize = df[df['SymbolMain'] == self.ASSET][['TickValue', 'ContractSize']].iloc[0]  

            if df.empty:
                raise Exception("DataFrame is empty")
                            
            return df

        except Exception as e:
            print(f"An error occurred: {e}")
            return None  


    def step(self, action) -> tuple[list, float, bool]:   
        if self.mode == 'train':
           if self.object_type == 'Multi_Instrument':                    
               if self.data[self.current_stock]['RoR'][self.current_step + self.k - 1] != self.r_ts[self.current_step + self.k - 1]:
                    raise AssertionError("Data do not fit!!")            
         
        self.current_step += 1
        if self.current_step == self.total_steps:
            self.is_terminal = True
        self.reward = (action-1) * self.r_ts[self.current_step + self.k - 1]
        self.state = tuple(self.r_ts[self.current_step:(self.k + self.current_step)])
        return self.state, self.reward, self.is_terminal
        
                                                       
    def step_eval(self, action = None, portfolio_history = None, debug = True) ->  tuple[float, bool]: 
        if self.current_step == 0:
            self.position_sizing = calculate_ex_ante_volatility(self.df, self.target_vola)
        
        if portfolio_history: 
            print(self.ticker)                                     
            timestamp, _, _, _ = portfolio_history[0]    
            idx = self.df.index.get_loc(timestamp)
            self.start_point = idx   

            self.Step = len(self.df) - self.start_point       
            init_net_worth = self.net_worth
                                      
            if self.Futures:
                for entry in portfolio_history:
                    _, _, _, action = entry           
                    try:
                        self._take_action_Futures(action, debug) 
                        self.net_worth = self.balance  
                        self.current_step += 1
                        reward = self.net_worth - init_net_worth     
                        done = self.net_worth <= 0 or self.current_step >= self.Step - 1
                        if done:    
                            break                                                                                   
                    except Exception as e:
                        print("An exception occurred during exec:", e)   
                return reward, done                                                       

        if action is not None:
            self.Step = len(self.df) - self.start_point
        
            init_net_worth = self.net_worth

            if action not in action_to_str:
                raise ActionInvalid(action)         

            if self.Futures:
                self._take_action_Futures(action, debug) 
                self.net_worth = self.balance 

            else:                          
                self._take_action_Stocks(action)
                self.net_worth = self.balance 

            self.current_step += 1
            reward = self.net_worth - init_net_worth     
            done = self.net_worth <= 0 or self.current_step >= self.Step - 1
            return reward, done  
        else:
            return 0, False                            


    def CalcReturnSeries(self, matchDate = None) -> pd.DataFrame:
        if matchDate is not None:     
            returns_data = [[matchDate, self.portfolio_history[0], 0, 0]]
            for i in range(1, len(self.portfolio_history)):
                previous_value = self.portfolio_history[i - 1]
                current_value = self.portfolio_history[i]
                numerical_difference = current_value - previous_value
                percentage_change = ((current_value / previous_value) - 1)

                date = self.df.index[i + self.k-1]  # Assuming dates are in the same order as portfolio_history
                returns_data.append([date, current_value, numerical_difference, percentage_change])
  
        else:
            returns_data = [[self.df.index[0], self.portfolio_history[0], 0, 0]] 
            for i in range(1, len(self.portfolio_history)):
                previous_value = self.portfolio_history[i - 1]
                current_value = self.portfolio_history[i]
                numerical_difference = current_value - previous_value
                percentage_change = ((current_value / previous_value) - 1)

                date = self.df.index[i]  # Assuming dates are in the same order as portfolio_history
                returns_data.append([date, current_value, numerical_difference, percentage_change])                  
            
        df = pd.DataFrame(returns_data, columns=['Date', 'Portfolio_Value', 'Numerical_Difference', 'RoR_PF'])    
        df['Positions'] =  self.position_history[:len(df)]    
        df['Actions'] =  [action_to_str.get(action, 'Unknown') for action in self.actions] + [pd.NA]  

        risk = RISK_metrics(return_series = df['RoR_PF'], env=self, total_contracts_sold = self.total_contracts_sold, 
                            total_contracts_bought=self.total_contracts_bought, actions = self.actions)
        self.risk_measures =  risk.Risk         
        return df      


    def __repr__(self):
        def mio_formatter(x):
                return f'{x / 1e6:.2f} Mio'        
        
        if hasattr(self, 'risk_measures') and self.append_position_tracking:
            profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE

            TC = sum(self.transactionCost)       

            format_action_count(0, self.actions)        

            table = f"""
            ------------------------------------------------------------------------------------------
            |{'Market':<30}|{self.cfg.data[self.ASSET]['Name'][0]:<57}|  
            |{' ':<30}|{self.ASSET:<57}|                   
            |{'Step':<30}|{self.current_step:<57}|
            |{'Balance':<30}|{round(self.balance, 4):<57}|
            |{'Shares held':<30}|{round(self.contracts_held, 4):<12} Total sold:   {str(round(self.total_contracts_sold, 4)): <30}|
            |{' ':<30}|{' ':<12} Total bought: {str(round(self.total_contracts_bought, 4)): <29} | 
            |{'Average abs position':<30}|{round(np.mean(self.append_position_tracking)):<8} {' ':<48}|                        
            |{' ':<30}|{' ':<12} {'---------': <43} |         
            |{'Decisions':30}|{' ':<12} Total BUY-actions:  {format_action_count(2, self.actions): <23} |        
            |{' ':<30}|{' ':<12} Total SELL-actions: {format_action_count(0, self.actions): <23} |     
            |{' ':<30}|{' ':<12} Total HOLD-actions: {format_action_count(1, self.actions): <23} |                 
            ------------------------------------------------------------------------------------------
            |{'Sum of Transaction costs':<30}|{round(TC, 4):<57}|
            |{'Initial captial':<30}|{mio_formatter(self.INITIAL_ACCOUNT_BALANCE):<57}|            
            |{'Net worth':<30}|{round(self.net_worth, 4):<20} Max net worth: {str(round(self.max_net_worth, 2)): <20} |
            |{'         ':<30}|{'':<20} Min net worth: {str(round(self.min_net_worth, 2)): <20} | 
            |{'Profit in Mio':<30}|{mio_formatter(round(profit, 4)):<57}|                        
            |{'Profit exact':<30}|{round(profit, 4):<57}|
            |{'Return':<30}|{round( ( (round(self.net_worth, 4)/self.INITIAL_ACCOUNT_BALANCE)-1) *100, 3):<8} % {' ':<46}|
            |{'Mean annualized return':<30}|{round(self.risk_measures['MeanReturn']*100, 3):<8} % {' ':<46}|           
            |{'Volatility (annualized)':<30}|{round(self.risk_measures['vola'],2):<8}  {' ':<47}| 
            |{'Risk free rate':<30}|{round(self.risk_measures['RF'],2):<3} % {' ':<51}|   
            |{'Value at Risk (95 %)':<30}|{round(self.risk_measures['VaR'],2):<8} {' ':<48}| 
            |{'Conditional VaR (ES)':<30}|{round(self.risk_measures['CVaR'],2):<8} {' ':<48}| 
            |{'Maximum Drawdown':<30}|{round(self.risk_measures['DD'],2):<16} {' ':<40}|
            |{'Sortino ratio':<30}|{round(self.risk_measures['SR'],2):<8}{' ':<49}|
            |{'Calmar ratio':<30}|{round(self.risk_measures['Calm'],2):<16} {' ':<40}| 
            |{'Tail ratio':<30}|{round(self.risk_measures['TailRatio'],2):<8} {' ':<48}|   
            |{'Downside risk':<30}|{round(self.risk_measures['DownSideRisk'],2):<8} {' ':<48}|    
            |{'Positive-to-Negative return':<30}|{round(self.risk_measures['PNRR'],2):<8} {' ':<48}|                                              
            |{'Sharpe ratio':<30}|{round(self.risk_measures['Sharp'],2):<8} {' ':<48}|                                                                                   
            ------------------------------------------------------------------------------------------
            """
            FileLogging(self.cfg, key = self.current_stock, content = table)            
            
            return table

        else:
            return "All positions were zero. No risk measures to compute."                    

    def _take_action_Stocks(self, action) -> None:
        
        current_point = self.current_step + self.start_point
        current_price = (self.df.loc[current_point, "high"] + self.df.loc[current_point, "low"]) / 2

        action_type = action
        self.actions.append(action)            
        amount = math.ceil(self.target_vola/self.df['close'].pct_change().std())       

        if action_type == 2:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = min(int(total_possible), amount)
            self.total_contracts_bought += shares_bought            
            prev_cost = self.cost_basis * self.contracts_held
            additional_cost = shares_bought * current_price
            self.transactionCost.append(additional_cost * self.TransactionCostFactor)            

            self.balance -= additional_cost
            self.contracts_held += shares_bought
            self.cost_basis = (prev_cost + additional_cost) / self.contracts_held if self.contracts_held != 0 else 0
            self.append_position_tracking.append(shares_bought)            

        elif action_type == 0:
            # Sell amount % of shares held
            shares_sold = min(int(self.contracts_held), amount)
            self.balance += shares_sold * current_price
            self.contracts_held -= shares_sold
            self.total_contracts_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
            additional_cost = shares_sold * current_price
            self.transactionCost.append(additional_cost * self.TransactionCostFactor)  
            self.append_position_tracking.append(shares_sold)                          

        self.net_worth = self.balance + self.contracts_held * current_price   
        self.position_history.append(self.contracts_held)     
        self.portfolio_history.append(self.net_worth)             

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.net_worth < self.min_net_worth:
            self.min_net_worth = self.net_worth              

        if self.contracts_held == 0:
            self.cost_basis = 0

    def _take_action_Futures(self, action, debug = True) -> None:    

        # https://www.cmegroup.com/education/courses/introduction-to-futures/calculating-futures-contract-profit-or-loss.html
        # findet zum zeitpunkt t statt
        # evaluieren der letzten Entscheidung

        current_point = self.current_step + self.start_point   
          
        try:
            Tick_Value = self.df['PointValue'][current_point] * self.df['TickSize'][current_point]              # Tick value  
            TickSize =  self.df['TickSize'][current_point]       # Tick size   
            PointValue =  self.df['PointValue'][current_point]   # Point Value
            if PointValue * TickSize != Tick_Value:
                raise AssertionError("Assertion Error: Tick size and Contract size mismatch")
      
        except AssertionError as e:
            print(e)

        current_price = self.df['AdjustedClose'][current_point]
        last_price = self.df['AdjustedClose'][current_point-1]

        profit_per_contract = (current_price - last_price)                            
        total_move = profit_per_contract * PointValue                                   # total move in ticks

        DayToDay_P_and_L = self.contracts_held * total_move                             # total profit in Fremdwährung
        DayToDay_P_and_L = DayToDay_P_and_L * self.df['Cross_Rate'][current_point]      # total profit in EUR       
        self.balance = self.balance + DayToDay_P_and_L  
        self.position_history.append(self.contracts_held)               

        # ----------------------
        #       Entscheidung wird für t + 1 getroffen
        action_type = action
        self.actions.append(action)   

        try:                                           
            positionSizing_2 = self.position_sizing['Position_Size'][current_point]
            POS = positionSizing_2               
       
        except ZeroDivisionError as zd_error:
            print(f"Error: {zd_error}")
                                  
        except Exception as e:
            print(f"An unexpected error occurred: {e}")                 
        
        # Transaction costs in EUR
        Potential_TC_per_contract = self.df['ContractValue_EUR'][current_point] * self.df['Cross_Rate'][current_point] * self.TransactionCostFactor  
             
        if action_type == 2:
            try:            
                total_possible_contracts = int(self.balance / (self.df['ContractValue_EUR'][current_point] * self.df['Cross_Rate'][current_point] * self.Margin_requirement) )
                contracts_bought = min(total_possible_contracts, POS)  
                self.total_contracts_bought += contracts_bought  

            except Exception as e:
                print(f"Error occurs in the buy action: {e}")                

            transaction_cost = contracts_bought * Potential_TC_per_contract
            self.transactionCost.append(transaction_cost)            

            if transaction_cost > self.balance:
                return      

            self.balance -= transaction_cost
            self.contracts_held += contracts_bought 
            self.append_position_tracking.append(contracts_bought)                    

        elif action_type == 0:
            contracts_sold = POS
            contracts_sold = min(POS, self.balance/Potential_TC_per_contract)             # Case if there is not enough Margin left            
            self.total_contracts_sold += contracts_sold

            # Calculate transaction cost based on margin
            transaction_cost = contracts_sold * Potential_TC_per_contract
            self.transactionCost.append(transaction_cost)              

            # Update margin balance
            self.balance -= transaction_cost
            self.contracts_held -= contracts_sold  
            self.append_position_tracking.append(contracts_sold)                                                                           

        if self.balance > self.max_net_worth:
            self.max_net_worth = self.balance

        if self.balance < self.min_net_worth:
            self.min_net_worth = self.balance     

        self.portfolio_history.append(self.balance)                   

        if debug and self.current_step > 1 and len(self.transactionCost) > 1:
            try:            
                print("=== Debug Information ===")
                print(f"Current Point: {current_point - self.start_point - 1}\n\n")

                print("=== Episode specific ===")                     
                print(f"Tick Size: {TickSize}")                
                print(f"Point Value: {self.df['PointValue'][current_point]}")   
                print(f"Value of One Tick: {Tick_Value}")               
                print(f"Profit per Contract: {round(profit_per_contract, 3)}")                                          
                print(f"Total Move in Cash: {round(total_move,3)}")
                print(f"Day-to-Day P&L: {round(DayToDay_P_and_L,3)}\n\n")
            
                print("=== Margin and action ===")             
                print(f"Margin Balance: {self.balance}")
                print(f"Margin Requirement: {self.Margin_requirement}")            
                print(f"Action Type: {self.actions[-2]}")
                print(f"Ordinary Sizing: {POS}")
                print(f"Potential TC per Lot: {round(Potential_TC_per_contract,3)}")
                print(f"Transaction Cost: {round(self.transactionCost[-2],3)}")
                print(f"Contracts Held After Action: {self.contracts_held}")
                print(f"Total Contracts Bought: {self.total_contracts_bought}")
                print(f"Total Contracts Sold: {self.total_contracts_sold}")
                print("==========================\n\n")   
            
            except ValueError as val_error:
                print(f"Unsupported Type error: {val_error}")      

            except Exception as e:
                print(f"An unexpected error occurred: {e}")                          
                