#%%

import gym


#%%

env = gym.make("Taxi-v2").env

"""
blue = passanger
purple = destination
yellew/red = empty taxi
green = full taxi
RGBY = location for destination and passanger
"""

env.render()
env.reset() # reset env and return initial state
#%%

print("State space: ",env.observation_space) # 500
print("Action space: ",env.action_space)     # 6

#%%
#taxi row,taxi column,passanger index,destination

state = env.encode(3,1,2,3)
print("State number: ",state)

env.s = state
env.render()

#%%
"""
Actions:
    There are 6 discrete deterministic actions:
        - 0: move south
        - 1: move north
        - 2: move east
        - 3: move west
        - 4: pickup passanger
        - 5: dropoff passanger
"""
# probability,next_state,reward,done
env.P[331]

#%%

# 1
env.reset()
time_step = 0
total_reward = 0
list_visualize = []
while True:
    
    time_step += 1
    
    # choose action
    action = env.action_space.sample()
    
    # perform action and get reward
    state,reward,done,info = env.step(action) #state = next_state
    
    # total reward
    total_reward += reward
    
    # visualize
    list_visualize.append({"frame":env.render(mode = "ansi"),
                           "state": state,
                           "action":action,
                           "reward":reward,
                           "total_reward":total_reward
            })
    #env.render()
    
    if done:
        break



#%%
import time
for i,frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Time Step:",i+1)
    print("State:",frame["state"])
    print("Action:",frame["action"])
    print("Reward:",frame["reward"])
    print("Total Reward:",frame["total_reward"])
    time.sleep(1)

#%%

































#%%