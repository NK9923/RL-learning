from utils.Configure import *
from utils.training import *
from utils.testing.testing import *
from utils.MACD.MACD import MACDStrategy
from utils.Plotting.plotting import *
from utils.LaTeX.generateTable import LaTeX
from utils.Neural_Networks.dqn import *

import argparse

def main(asset_Class, sett, N, Algo = "DQN", modelPath = None, debug=False, individual = None, multitrain = False, Train = True, eval = False) -> None:
        
    cfg = Config(create_output_dir = modelPath, algo_name = Algo)
    cfg.update_settings(asset_Class=asset_Class, Setting = sett, N_episodes = N)      
    cfg.getTrainAndTestData()     

    if Train:      
        try:        
            print(cfg.DATES[sett])               
            if individual is None and multitrain is True:                                                      
                env, agent = env_agent_config(cfg.train_data, cfg, 'train', individual_ticker = None)                             

                # ===== Training process (random sampling from ticker list) =====
                train(cfg, env, agent, early_stopping = early_stopping)  
                agent.save(path=cfg.model_path)  
                print(agent.memory_tracker)    

            elif multitrain is False and individual is not None:
                print(f'\nTraining on individual stocks: {cfg.train_eps} Episodes\n')  
                inst = individual                                
                       
                print(f'Evaluation for: {inst} | {asset_Class}')                  
                env, agent = env_agent_config(cfg.train_data[inst]['RoR'].tolist(), cfg, 'train', individual_ticker = inst)              

                # ===== Training process (on particular predefined market) ======
                train(cfg, env, agent)
                agent.save(path=cfg.model_path) 
                print(agent.memory_tracker) 

            return True

        except Exception as e:
            print(f"An error occurred during training: {e}")
            return False  

    elif Train is False and eval is False:
        return True                                 
        
    if eval: 

        PF = {} 
        History = {}   
        Risk_factors = {}
        algo_portfolio_performance = {}                 
        
        if not cfg.check_paths_exist():
            raise Exception("Training was not performed for both algorithms. Models do not exist.")      

        else:    
            for algo in ['DQN', 'DDQN']:
                cfg.update_settings(ALGO=algo)

                algo_PF = {}
                algo_MACD = {} 
                BH_PF = {} 
                BH_Risk_factors = {}                                              
                algo_History = {}
                algo_Risk_factors = {}
                algo_MACD_risk_factors = {}            

                for inst in cfg.asset_class_Ticker:
                    if individual is not None:
                        inst = individual

                    cfg.getTrainAndTestData(inst)
                    env, agent = env_agent_config(cfg.test_data['RoR'].tolist(), cfg, 'test', individual_ticker=inst)
                    agent.load(path=cfg.model_path)

                    # testing
                    env.reset()
                    portfolio_history, _ = test(cfg, env, agent)

                    # MACD strategy execution
                    env.reset()
                    macd = MACDStrategy(envir=env, config=cfg, slow=26, fast=12, smooth=9, Current_market=inst)
                    algo_MACD_risk_factors[inst] = macd.Risk_measures_MACD
                    algo_MACD[inst] =  macd.MACD_porfolio_value  

                    # Buy and Hold strategy                    
                    BH_PF[inst], BH_Risk_factors["BuyAndHold_" + inst] = Calculate_Buy_and_Hold(cfg, env)                                      

                    # Reinforcement Learning strategy (DQN or DDQN)
                    env.reset()
                    _, done = env.step_eval(portfolio_history=portfolio_history, debug=debug)

                    if done:
                        algo_PF[inst] = env.CalcReturnSeries(portfolio_history[0][0])
                        algo_Risk_factors[inst] = env.risk_measures
                        Market_history = pd.DataFrame(portfolio_history, columns=['Date', 'Value', 'Action', 'Type'])
                        Market_history['Positions'] = env.position_history[1:]
                        algo_History[inst] = Market_history

                # == Portfolio Kennzahlen für MACD und Buy and Hold =====   
                algo_portfolio_performance[algo], algo_Risk_factors["Portf"] = build_portfolio(env, cfg, algo_PF)    
                MACD_portfolio_performance, Risk_factors_MACD = build_portfolio(env, cfg, algo_MACD)    
                BH_portfolio_performance, Risk_factors_BH = build_BH_portfolio(env, cfg, BH_PF)                                    

                # Store results in the corresponding dictionaries
                PF[algo] = algo_PF            
                History[algo] = algo_History
                Risk_factors[algo] = algo_Risk_factors            


                # == LaTeX Tabellen ausgeben == 
                if generateLatex:

                    L = LaTeX(Risk_factors[algo].keys(), OutputPATH = cfg.Output_PATH, 
                              Asset_class=asset_Class, algo = cfg.algo_name, test_year = cfg.TestinDate[2][0:4], 
                              Name_dict = cfg.full_names)
                
                    L.Fill_Table(Risk_factors[algo]) 

                    L.Fill_Table({'BuyAndHold': Risk_factors_BH}) 
                    L.Fill_Table({'MACD': Risk_factors_MACD})                                 
                    L.saveTable()                  

                if representation:

                    Plotting_1(cfg=cfg, PATH=cfg.Output_PATH, data = algo_portfolio_performance[algo], 
                               asset_Class = asset_Class, settings = sett, Transaction_History = History[algo], individual = PF[algo])   
                                         
            # # == Plotting =================    
            if representation:
                Plotting_2(cfg=cfg, data = algo_portfolio_performance, 
                           asset_Class = asset_Class, settings = sett, Transaction_History = History, individual = PF, 
                           MACD = MACD_portfolio_performance, Risk_MACD = algo_MACD_risk_factors,                                    # MACD components 
                           BH = BH_portfolio_performance, Risk_BH = BH_Risk_factors,                                                 # Buy and Hold components   
                           Risk_RL = Risk_factors)                                                                                   # Risk metrics DQN and DDQN  

            if representation :
                return (Risk_factors, Risk_factors_BH, Risk_factors_MACD)     

            if individual is not None: 
                del agent, env, cfg                                   
                return              
                          
    else:
        raise NotImplementedError()  
      
    del agent, env, cfg  
                                                                                                                                 

if __name__ == "__main__": 

    generateLatex = False   
    early_stopping = False   
    representation = False               
    
    Future_classes = ['FX','Equities', 'Bonds', 'Commodities_Energies', 'Commodities_Metals', 'Commodities_Softs']    
    Train_eps = {'FX' : 18, 'Equities' : 18, 'Bonds' : 36 , 'Commodities_Energies' : 18 , 'Commodities_Metals' : 24 , 'Commodities_Softs' : 40}     
    
    parser = argparse.ArgumentParser(description='Master Thesis Reinforcement Learning')
    parser.add_argument('--models', default=None, help='Specify the Output Folder name')   
    parser.add_argument('--Ticker', default=None, help='Should an individual market be trained. Requires Ticker')      
  
    args = parser.parse_args()
    models = args.models    
    Ticker = args.Ticker   

    Train = True if models is None else False
    Multi = False if Ticker is not None else True
    dir_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S") if models is None else models
                                   
    try:
        for asset_Class in Future_classes: 
            N_eisodes = Train_eps[asset_Class]
            
            asset_class_results = {}                       
            for Setting in range(4):   
                if Train:      
                    # Training loop                                               
                    main(asset_Class, sett=Setting, N=N_eisodes, Algo = "DQN", modelPath = dir_name, debug = False, individual = Ticker, multitrain=Multi, Train = True)                              
                    main(asset_Class, sett=Setting, N=N_eisodes, Algo = "DDQN", modelPath = dir_name, debug = False, individual = Ticker, multitrain=Multi, Train = True) 
                else:     
                    # Training loop                                                                   
                    results = main(asset_Class, sett=Setting, N=N_eisodes, Algo = "DQN", modelPath = dir_name, debug = False, individual = Ticker, multitrain=Multi, Train = False, eval = True)   
                    asset_class_results[Setting] = results 
 
            if representation and models is not None:           
                abs_ret(asset_class_results, asset_Class, models)
                                                                                                                         
    except KeyboardInterrupt:
        print("Aborted")  

    print("Finished")                                                       