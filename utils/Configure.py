import os
import yaml
import torch
import json
import pandas as pd

from utils.trading_env.trading_env_test import TradingEnv
from utils.datadownload import getData_from_csv
from utils.Neural_Networks.dqn import DQN

class Config:
    class Algorithm:
        DQN = "DQN"
        DDQN = "DDQN"

    def __init__(self, create_output_dir, asset_Class = 'FX', Setting = 0, algo_name=None, config_file_path="config/config.yml"):
        self.enable_repr = False        
        self._load_config(config_file_path)
        self.Bloomberg_Ticker = self._load_bloomberg_ticker()
        self._set_defaults(existing_path = create_output_dir, ALGO = algo_name)
        self.update_settings(asset_Class = asset_Class, Setting=Setting)


    def _load_config(self, config_file_path) -> None:
        try:
            with open(os.path.join(os.getcwd(), config_file_path), 'r') as stream:
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{config_file_path}' not found.")


    def _load_bloomberg_ticker(self) -> None:
        try:
            with open('Input/jsonFiles/Symbols.json', 'r') as json_file:
                self.bloomberg_ticker = json.load(json_file)
            return self.bloomberg_ticker                
        except FileNotFoundError:
            raise FileNotFoundError("Bloomberg ticker file 'Input/jsonFiles/Symbols.json' not found.")


    def _set_defaults(self, existing_path, ALGO) -> None:
        if ALGO is not None:
            self.algo_name = ALGO
        else:
            self.algo_name = Config.Algorithm.DQN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_time = existing_path
        self.train_eps = self.config.get('train_eps', 37)
        self.state_space_dim = self.config.get('state_space_dim', 50)
        self.action_space_dim = self.config.get('action_space_dim', 3)
        self.gamma = self.config.get('gamma', 0.95)
        self.tau = 0.01
        self.softupdate = False
        self.epsilon_start = self.config.get('epsilon_start', 0.9)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 500)
        self.lr = self.config.get('lr', 0.0001)
        self.memory_capacity = self.config.get('memory_capacity', 1000)
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update = self.config.get('target_update', 4)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.DATES = self.config['Dates']
        self.TYPE = self.config['Input_Type']        
        self.train_data = pd.DataFrame() 
        self.test_data = pd.DataFrame()                             


    def update_settings(self, ALGO = None, asset_Class = None, Setting = None, N_episodes = None, learningRate = None) -> None:
        if ALGO is not None:
            self.algo_name = ALGO
                    
        if asset_Class is not None: 
            if not hasattr(self, 'asset_Class'):
                self.asset_Class = asset_Class
            else:  
                if self.asset_Class != asset_Class: 
                    self.asset_Class = asset_Class
                                                   
            self.asset_class_Ticker = self.bloomberg_ticker.get(asset_Class, [])
            data, _ = getData_from_csv(self.asset_class_Ticker, full_data = False)
            self.data = data  
                          
        if Setting is not None:            
            self.Setting = Setting
            self.TestinDate = self.config['Dates'][Setting]
            
        if self.asset_Class is not None and self.Setting is not None:
            self._create_output_directories() 
            
        if N_episodes is not None:
            self.train_eps = N_episodes   
              
        if learningRate is not None:
            self.lr = learningRate   

        self._create_output_directories()                                                    


    def getTrainAndTestData(self, key = None):
        start_date = pd.to_datetime(self.TestinDate[1], format='%Y-%m-%d') #- pd.DateOffset(days=self.state_space_dim)
        end_date = pd.to_datetime(self.TestinDate[2], format='%Y-%m-%d')
        self.full_names = {}        
        
        if key is not None:        
            try:     
                data = self.data[key]
                self.train_data = data.loc[self.TestinDate[0]:self.TestinDate[1]][self.TYPE]
                self.test_data =  data[(data.index >= start_date) & (data.index <= end_date)][self.TYPE]                   
            except KeyError:      
                raise KeyError("Ticker is not in the asset class. First set asset class")
        else:
             data = self.data   
             self.train_data = {key: df.loc[self.TestinDate[0]:self.TestinDate[1]][self.TYPE] for key, df in data.items()}

             self.test_data = {}
             for key, df in data.items():
                 selected_data = df[(df.index >= start_date) & (df.index <= end_date)][self.TYPE]
                 self.test_data[key] = selected_data
             
        for key in self.data.keys():
            self.full_names[key] = self.data[key]['Name'][0]                                     


    def _create_output_directories(self) -> None:
        model_dir = os.path.join(os.getcwd(), "outputs", self.curr_time)
        DQN_model_path = os.path.join(model_dir, f'models_DQN_{self.asset_Class}{self.TestinDate[2][0:4]}')
        DDQN_model_path = os.path.join(model_dir, f'models_DDQN_{self.asset_Class}{self.TestinDate[2][0:4]}')  
        self.PATHS = [DQN_model_path.replace("models", 'results'), DDQN_model_path.replace("models", 'results')]              
        
        if self.algo_name == Config.Algorithm.DQN:
            self.model_path = DQN_model_path
            self.Output_PATH = f'{os.path.split(self.model_path)[0]}\\results_DQN_{self.asset_Class}{self.TestinDate[2][0:4]}'            
        elif self.algo_name == Config.Algorithm.DDQN:
            self.model_path = DDQN_model_path
            self.Output_PATH = f'{os.path.split(self.model_path)[0]}\\results_DDQN_{self.asset_Class}{self.TestinDate[2][0:4]}'            

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.Output_PATH):         
            os.makedirs(self.Output_PATH)  

        self._SaveSpec()      


    def check_paths_exist(self) -> bool:
        missing_paths = [path for path in self.PATHS if not os.path.exists(path)]     
        print(missing_paths)           
        return all(os.path.exists(path) for path in self.PATHS)                       


    def _SaveSpec(self) -> None:
        with open('Input/jsonFiles/spec_parameters.json', 'r') as json_file:
            spec_parameters = json.load(json_file)   

        for key, _ in spec_parameters.items():
            value = getattr(self, key)            
            spec_parameters[key] = value

        PATH = self.model_path + '/spec_parameters.json'
        with open(PATH, 'w') as json_file:
            json.dump(spec_parameters, json_file, indent=4)             
     

    def __repr__(self) -> None:
        if not self.enable_repr:
            self.enable_repr = True            
            return "Config object created"
        
        attributes = [
            "algo_name", "device", "train_eps",
            "state_space_dim", "gamma", "lr", "memory_capacity", "batch_size",
            "epsilon_start", "epsilon_end", "epsilon_decay",
            "asset_Class", "asset_class_Ticker", "TestinDate",
            "target_update", "hidden_dim"
        ]

        repr_string = "\nConfig(\n    " + ",\n    ".join([f"{attr}={getattr(self, attr)}" for attr in attributes]) + "\n)\n"
        return repr_string                      
     
                                   
def env_agent_config(data, cfg, mode, individual_ticker = None):
    if mode == 'train':
        env = TradingEnv(cfg=cfg,  
                         returns_data=data, 
                         k_value=cfg.state_space_dim, 
                         mode=mode, 
                         key = individual_ticker, 
                         df = cfg.train_data)
    if mode == 'test':
        env = TradingEnv(cfg=cfg,  
                         returns_data=data, 
                         k_value=cfg.state_space_dim, 
                         mode=mode, 
                         key = individual_ticker, 
                         df = cfg.test_data)          
              
    agent = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg)      
    return env, agent


