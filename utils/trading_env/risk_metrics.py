from empyrical import annual_return, value_at_risk, conditional_value_at_risk, calmar_ratio, downside_risk, sharpe_ratio, sortino_ratio, tail_ratio
from empyrical.stats import annual_volatility 

import numpy as np
import pandas as pd

# Some one risk metrics functions have been written. However, I mostly used the ones from the empyirical package. The reason for that is, that they showed rather slight variations
# to my functions, but I guess they are well tested and rather well documented. So these implementations are correct. 


class RISK_metrics:
    def __init__(self, return_series, data = None, env = None, total_contracts_sold = None, 
                 total_contracts_bought = None, actions = None, InitialCapital = None):
        self.envir = env         
        if env is not None:
            self.PF_history = self.envir.portfolio_history 
            self.Risk_free_IR = self.envir.rf   
            self.PF_TC = self.envir.transactionCost  
            self.starting_value = self.envir.INITIAL_ACCOUNT_BALANCE  
                                          
        else:
            try:            
                self.PF_history = data['Portfolio_Value'].tolist()
                self.starting_value = data['Portfolio_Value'][0]                
            except Exception:
                 self.PF_history = (1 + data['LogRoR'][1:].cumsum()) * InitialCapital        
                 self.PF_history = self.PF_history.tolist()
                 self.starting_value = InitialCapital                                                            
                
            self.Risk_free_IR = 1.0 
            self.PF_TC = 0

        self.total_contracts_sold  = total_contracts_sold    
        self.total_contracts_bought = total_contracts_bought      
        self.actions = actions              
        self.confidence_level = 0.95
        self.return_series = return_series                 
        self.Risk = self.CalcMeasures()            

    def treasury_bond_daily_return_rate(self) -> float:
        r_year = self.Risk_free_IR / 100 
        return (1 + r_year)**(1 / 365) - 1        
                        
    # == Sharpe Ratio =================
    def sharpe_ratio_1(self, N=252) -> float:    
        return_series = np.array(self.return_series)
        excess_return_series = return_series - self.treasury_bond_daily_return_rate()
        mean =  np.nanmean(excess_return_series)
        sigma =  np.nanstd(excess_return_series)       
        if sigma == 0:
            return 0
        else:
            return (mean / sigma)  * np.sqrt(N)   

    # == Sortino ratio ========================
    def sortino_ratio(self, N=252) -> float:
        return_series = np.array(self.return_series)
        if np.all(return_series >= 0):
            return 0.0
        mean = return_series.mean() * np.sqrt(N) - self.Risk_free_IR / 100
        std_neg = return_series[return_series < 0].std() * np.sqrt(N)
        if std_neg == 0 or np.isnan(std_neg) or np.isinf(std_neg):
            return 0.0
        else:
            return mean / std_neg

    # == Max Drawdown =========================
    def max_drawdown(self) -> float:
        df = pd.DataFrame(self.return_series)
        comp_ret = (df + 1).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()[0]

    # == Transaction Cost =====================
    def calculate_total_transaction_costs(self) -> float:
        total_costs = sum(self.PF_TC)
        return total_costs

    # == Conditional Value-at-Risk (CVaR) ====
    def ExpectedShortFall(self) -> float:
        data_sorted = np.sort(self.return_series)
        index = int((1 - self.confidence_level) * len(data_sorted))
        tail_losses = data_sorted[index:]
        expected_shortfall = np.mean(tail_losses)
        return expected_shortfall

    # == Calculate all Risk Measures ===========
    def CalcMeasures(self) -> None:
        if not self.envir is None:
            if self.envir.TransactionCostFactor:
                TransKosten = self.calculate_total_transaction_costs()
        else:
            TransKosten = 0

        returns = np.array(self.return_series)

        if self.actions is not None:
            buy_actions_count = len([action for action in self.actions if action == 2])
        else:
            buy_actions_count = 0     

        if self.actions is not None:
            sell_actions_count = len([action for action in self.actions if action == 0])
        else:
            sell_actions_count = 0  

        if self.actions is not None:
            hold_actions_count = len([action for action in self.actions if action == 1])
        else:
            hold_actions_count = 0   

        self.sharpe_ratio_1()                                           

        try:
            metrics = {
                'IC': self.starting_value,  
                'MaxVal': max(self.PF_history),
                'MinVal': min(self.PF_history),                                           
                'abs_g': self.PF_history[-1] - self.starting_value,
                'TotalBUY' : self.total_contracts_bought or 0,
                'TotalSELL' :  self.total_contracts_sold or 0,
                'BUY_actions' : buy_actions_count,        
                'SELL_actions' : sell_actions_count, 
                'HOLD_actions' : hold_actions_count,                                                                                   
                'abs_ret': (self.PF_history[-1] / self.starting_value) -1,
                'MeanReturn' :  annual_return(returns),               
                'vola': annual_volatility(returns),
                'PV': self.PF_history[-1],
                'TC': TransKosten,
                'RF': self.Risk_free_IR,                
                'VaR': value_at_risk(self.return_series, cutoff=0.05) * np.sqrt(252),
                'CVaR': conditional_value_at_risk(self.return_series, cutoff=0.05) * np.sqrt(252),
                'DD': self.max_drawdown(),
                'SR': sortino_ratio(returns, required_return = self.treasury_bond_daily_return_rate(), period='daily'),
                'Sharp': sharpe_ratio(returns, risk_free=self.treasury_bond_daily_return_rate(), period='daily'),                
                'Calm': calmar_ratio(returns),
                'TailRatio'  : tail_ratio(returns),
                'DownSideRisk' : downside_risk(returns, required_return=1, period='daily'),
                'PNRR' :  len(self.return_series[self.return_series > 0]) / len(self.return_series[self.return_series < 0])          # "Positive-to-Negative Return Ratio."                  
            }
            return metrics
        except Exception as e:
            print(f"Error in computing the risk measures: {e}")
            metrics = {
                'IC': 0,  'MaxVal': 0,'MinVal': 0, 'abs_g': 0,'TotalBUY' : 0, 'TotalSELL' :  0, 'BUY_actions' : 0, 'SELL_actions' : 0, 'HOLD_actions' : 0,                                                                 
                'abs_ret': 0, 'MeanReturn' :  0, 'vola': 0, 'PV': 0, 'TC': 0, 'RF': 0, 'VaR': 0, 'CVaR': 0, 'DD': 0, 'SR': 0, 'Sharp': 0,                 
                'Calm': 0, 'TailRatio'  : 0, 'DownSideRisk' : 0, 'PNRR' : 0                     
            }            
            return metrics










        # metrics = {
        #     'IC': self.starting_value,  
        #     'MaxVal': max(self.PF_history),
        #     'MinVal': min(self.PF_history),                                           
        #     'abs_g': self.PF_history[-1] - self.starting_value,
        #     'TotalBUY': 0,
        #     'TotalSELL': 0,
        #     'BUY_actions': 0,
        #     'SELL_actions': 0,
        #     'HOLD_actions': 0,
        #     'abs_ret': 0,
        #     'MeanReturn': 0,
        #     'vola': 0,
        #     'PV': self.PF_history[-1],
        #     'TC': TransKosten,
        #     'RF': self.Risk_free_IR,
        #     'VaR': 0,
        #     'CVaR': 0,
        #     'DD': 0,
        #     'SR': 0,
        #     'Sharp': 0,
        #     'Calm': 0,
        #     'TailRatio': 0,
        #     'DownSideRisk': 0,
        #     'PNRR': 0
        # }

        # try:
        #     # Update the metrics dictionary with calculated values
        #     metrics.update({
        #         'TotalBUY': self.total_contracts_bought,
        #         'TotalSELL': self.total_contracts_sold,
        #         'BUY_actions': buy_actions_count,
        #         'SELL_actions': sell_actions_count,
        #         'HOLD_actions': hold_actions_count,
        #         'abs_ret': (self.PF_history[-1] / self.starting_value) - 1,
        #         'MeanReturn': annual_return(returns),
        #         'vola': annual_volatility(returns),
        #         'VaR': value_at_risk(self.return_series, cutoff=0.05) * np.sqrt(252),
        #         'CVaR': conditional_value_at_risk(self.return_series, cutoff=0.05) * np.sqrt(252),
        #         'DD': self.max_drawdown(),
        #         'SR': sortino_ratio(returns, required_return=self.treasury_bond_daily_return_rate(), period='daily'),
        #         'Sharp': sharpe_ratio(returns, risk_free=self.treasury_bond_daily_return_rate(), period='daily'),
        #         'Calm': calmar_ratio(returns),
        #         'TailRatio': tail_ratio(returns),
        #         'DownSideRisk': downside_risk(returns, required_return=1, period='daily'),
        #         'PNRR': len(self.return_series[self.return_series > 0]) / len(self.return_series[self.return_series < 0])
        #     })
        # except Exception as e:
        #     print(f"Error in computing the risk measures: {e}")
        # return metrics
                                   