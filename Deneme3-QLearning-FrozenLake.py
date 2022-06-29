#%%

import gym
import numpy as np
import matplotlib.pyplot as plt
import random

#%%

env = gym.make("FrozenLake-v0",is_slippery=False).env

#%%
"""from gym.envs.registration import register
register(
        id="FrozenLakeNotSlippery-v0",
        entry_point = "gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"map_name":"4x4","is_slippery":False},
        max_episode_steps=100,
        reward_threshold=0.78)"""




# Q Table

#q_table = np.zeros([state,action])
q_table = np.zeros([env.observation_space.n,env.action_space.n])

# Hyperparameter
alpha = 0.8
gamma = 0.95
epsilon = 0.1


#%%

# Plotting Matrix
reward_list = []
dropout_list = []


episode_number = 10000

for i in range(1,episode_number):
    
    #initialize enviroment
    state = env.reset()
    
    reward_count = 0 
    dropouts = 0
    while True:
        # exploit vs explore to find action
        # %10 = explore, %90 = exploit
        
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # action process and take reward/observation
        next_state,reward,done,_ = env.step(action)
        while state == next_state:
            next_state,reward,done,_ = env.step(env.action_space.sample())
        #env.render()
        # Q learning function
        
        old_value = q_table[state,action] # old_value
        next_max = np.max(q_table[next_state]) # next_max
        
        if reward == 1:
            reward = 2
        elif reward == 0 and done == True:
            reward = -10

        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # Q table update
        
        q_table[state,action] = next_value
        
        # update state
        
        state = next_state
        
        # find wrong dropouts
        
        if reward == -10 and done == True:
            dropouts += 1
            
        reward_count += reward
        
        if done:
            break
    
    if i % 10 == 0:
        dropout_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode: {}, Reward: {}, Wrong Moves: {}".format(i,reward_count,dropouts))





#%%

plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
plt.plot(dropout_list)
plt.xlabel("Episode")
plt.ylabel("Dropout")
plt.show()


#%%











































#%%



#%%

















































#%%