from collections import Counter 
import pandas as pd
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm 

def Plot_Learning(rewards, ma_rewards, outputPath, window_size=10) -> None:
    rewards = pd.Series(rewards)
    ma_rewards = pd.Series(ma_rewards)
    rolling_std_rew = rewards.rolling(window=window_size).std()    
    rolling_std = ma_rewards.rolling(window=window_size).std()

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, facecolor='lightgray')
    ax.grid(True, linestyle='--', color='gray')
    plt.plot(rewards, label='Rewards', color='deeppink')
    plt.fill_between(range(len(ma_rewards)), rewards + rolling_std_rew, rewards - rolling_std_rew, color='deeppink', alpha=0.3)     
    plt.plot(ma_rewards, label=f'Moving Average ({window_size}-period)', color='green')
    plt.fill_between(range(len(ma_rewards)), ma_rewards + rolling_std, ma_rewards - rolling_std, color='green', alpha=0.3)

    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend()
    plt.title("Learning process")   
    plt_name = "training_process" + str(len(rewards)) + ".pdf"     
    plt.savefig(os.path.join(outputPath, plt_name), format='pdf')       


def train(cfg, env, agent, early_stopping = False) -> None:
    print('Start Training!')
    rewards = []
    ma_rewards = []
    used_tickers = []
    
    EPSILON = 1e-6   
 
    for i_ep in tqdm(range(cfg.train_eps), total=cfg.train_eps, leave=True, desc='Training Progress'):
             
        ep_reward = 0
        state = env.reset()
        used_tickers.append(env.ticker)            
                
        actions = []
        avg_loss = []
        mean_loss = []                
        while True:
            action = agent.choose_action(state)
            actions.append(action)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()         
            ep_reward += reward
          
            if len(agent.memory) > agent.batch_size:
                #agent.update()
                loss_np = agent.loss.cpu().detach().numpy()                
                avg_loss.append(loss_np)
                if not mean_loss:
                    mean_loss.append(avg_loss[-1])                                                     
                mean_loss.append(0.9 * mean_loss[-1] + 0.1 * avg_loss[-1])      

            # early stopping procedure
            if len(mean_loss) > 100:
                if early_stopping and abs(mean_loss[-1]-mean_loss[-100]) < EPSILON:                   
                    done = True                     

            if done:
                break
            
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
            
        if (i_ep + 1) % 10 == 0:
            print('\nEpisode:{}/{}, Reward:{}'.format(i_ep + 1, cfg.train_eps, ep_reward))

    print('Finish Training')
    ticker_counts = list(Counter(used_tickers).items())

    Plot_Learning(rewards, ma_rewards, cfg.Output_PATH) 

    for ticker, count in ticker_counts:
        print(f'Ticker: {ticker}, Count: {count}') 

