from utils.Configure import *

import gym
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils.datadownload import *

Asset = 'CL'
cfg = Config()
cfg.update_settings('Commodities')
cfg.getTrainAndTestData(Asset) 


def test_trading_env():

    env, agent = env_agent_config(cfg.train_data['RoR'].tolist(), cfg, 'train', individual_ticker = Asset)   
    env.reset()

    # Define the number of episodes
    num_episodes = 1

    for episode in range(num_episodes):
        done = False
        total_reward = 0

        while not done:
            # Replace this with your RL agent's action selection logic
            action = random.randint(0, 2)  # Random action for testing

            # Take a step in the environment
            _, done = env.step_eval(action)
            if done:
                Ret = env.CalcReturnSeries()                            

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        print(env)       



def test_config_class():
    
    # Initialize the Config object
    print("Initialized Config object:")
    config = Config('FX', 1)
    
    # Test changing asset class
    print("Config after changing asset class:")
    config.update_settings('Equities')
    
    # Test changing setting
    print("Config after changing setting:")
    config.update_settings(Setting=2)
    

    # Test changing learning rate
    print("Config after changing learning rate:")
    config.update_settings(learningRate=0.001)
    

    # Test changing number of episodes
    print("Config after changing number of episodes:")
    config.update_settings(N_episodes=50)

    # Test saving spec parameters
    config.SaveSpec()
    print("Spec parameters saved to file.")


if __name__ == "__main__":

    test_trading_env()