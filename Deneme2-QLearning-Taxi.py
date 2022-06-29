#%%

import gym
import numpy as np
import matplotlib.pyplot as plt
import random

#%%

env = gym.make("Taxi-v2").env

#%%

# Q Table

#q_table = np.zeros([state,action])
q_table = np.zeros([env.observation_space.n,env.action_space.n])

# Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1


#%%

# Plotting Matrix
reward_list = []
dropout_list = []


episode_number = 100000

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
        
        # Q learning function
        
        old_value = q_table[state,action] # old_value
        next_max = np.max(q_table[next_state]) # next_max

        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # Q table update
        
        q_table[state,action] = next_value
        
        # update state
        
        state = next_state
        
        # find wrong dropouts
        
        if reward == -10:
            dropouts += 1
            
        reward_count += reward
        
        if done:
            break
    
    if i % 10 == 0:
        dropout_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode: {}, Reward: {}, Wrong Dropout: {}".format(i,reward_count,dropouts))





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