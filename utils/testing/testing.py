from utils.Plotting import *
from utils.trading_env.risk_metrics import RISK_metrics
from utils.Logging.Log import *
import matplotlib.pyplot as plt 
import pandas as pd


def Calculate_Buy_and_Hold(cfg, env):
    ReferenceTS = (1 + cfg.test_data['LogRoR'][1:].cumsum()) *  env.INITIAL_ACCOUNT_BALANCE             # Buy and Hold portfolio_value
    value_to_add = pd.Series(env.INITIAL_ACCOUNT_BALANCE, index=[cfg.test_data.index[0]])
    combined_series = pd.concat([value_to_add, ReferenceTS])
    
    RM_1 = RISK_metrics(return_series = cfg.test_data['RoR'], data = cfg.test_data, InitialCapital= env.INITIAL_ACCOUNT_BALANCE)    
    return combined_series, RM_1.Risk,    


def build_BH_portfolio(env, cfg, BH_PF):
    Date = BH_PF[list(BH_PF.keys())[0]].index     
    combined_df = pd.concat(list(BH_PF.values()), axis=1, keys=BH_PF.keys())  
    combined_df.fillna(method='ffill', inplace=True)
    mean_rowwise = combined_df.mean(axis=1)      
    portfolio_performance = pd.DataFrame({'Portfolio_Value': mean_rowwise})
       
    RM = RISK_metrics(return_series = portfolio_performance['Portfolio_Value'].pct_change()[1:], data = portfolio_performance)    
    FileLogging(cfg, 'Buy_and_Hold', RM.Risk) 
    return portfolio_performance, RM.Risk      


def build_portfolio(env, cfg, Portfolio_history):
    date = Portfolio_history[list(Portfolio_history.keys())[0]]['Date']  
    combined_df = pd.concat(list(Portfolio_history.values()), axis=1, keys=Portfolio_history.keys())
    numerical_difference_columns = [col for col in combined_df.columns if 'Numerical_Difference' in col]
    Pf_values = [col for col in combined_df.columns if 'Portfolio_Value' in col]    
    row_sums = combined_df[numerical_difference_columns].sum(axis=1)

    scaling_factor = 1/len(cfg.test_data.keys())      
    scaled_series = row_sums * scaling_factor  
    Pf_values = combined_df[Pf_values] 
    Pf_values.columns = ['_'.join(col).strip() for col in Pf_values.columns.values]    
    
    portfolio_performance = pd.concat([date, Pf_values, pd.DataFrame({'Combined': scaled_series})], axis=1)

    portfolio_value = [env.INITIAL_ACCOUNT_BALANCE]  # Initialize with the starting value
    for i in range(1, len(portfolio_performance)):
        previous_value = portfolio_value[i - 1]
        performance = portfolio_performance['Combined'][i]
        current_value = previous_value + performance
        portfolio_value.append(current_value)    

    portfolio_performance['Portfolio_Value'] = portfolio_value   
    portfolio_performance = portfolio_performance[(portfolio_performance['Date']>= cfg.TestinDate[1]) & (portfolio_performance['Date'] < cfg.TestinDate[2])]
      
    RM = RISK_metrics(return_series = portfolio_performance['Portfolio_Value'].pct_change()[1:], data = portfolio_performance)      
    FileLogging(cfg, 'Portfolio', RM.Risk)  
      
    return portfolio_performance, RM.Risk


def test(cfg, env, agent):
    print('Start Testing!')

    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0

    history = []    

    stocks = env.ticker
    rewards = []  # record total rewards
    if env.object_type == 'Multi_Instrument':
        for i_ep in range(len(stocks)):
            ep_reward = 0
            state = env.reset()
            actions = []
            while True:
                action = agent.choose_action(state)
                actions.append(action)
                next_state, reward, done = env.step(action)
                state = next_state
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)
            print(f"Episode:{i_ep + 1}/{len(stocks)}，Reward:{ep_reward:.1f}")
    else:
        ep_reward = 0
        cumulative_rewards = []
        state = env.reset()
        actions = []
        t = 0
        
        while True:
            action = agent.choose_action(state)
            actions.append(action)    
                          
            # BUY
            if action == 2:
                history.append((cfg.test_data.index[t + env.k - 1], cfg.test_data['Adjusted_Close_EUR'][t + env.k - 1], "BUY", 2))           
       
            # SELL
            elif action == 0:             
                history.append((cfg.test_data.index[t + env.k - 1], cfg.test_data['Adjusted_Close_EUR'][t + env.k - 1], "SELL", 0)) 
            # HOLD
            else:
                history.append((cfg.test_data.index[t + env.k - 1], cfg.test_data['Adjusted_Close_EUR'][t + env.k - 1], "HOLD", 1))
            
            t += 1
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            cumulative_rewards.append(ep_reward)
            if done:
                break

    print('Finish Testing!')
    return history, ep_reward

