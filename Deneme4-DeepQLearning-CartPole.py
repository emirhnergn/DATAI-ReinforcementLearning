#%%

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam
import random
import time
from warnings import filterwarnings
filterwarnings("ignore")


#%%

class DQLAgent:
    
    def __init__(self,env):
        
        # hyperparameter/parameter
        self.state_size  = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1 # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
    
    def build_model(self):
        
        # neural network for deep Q learning
        model = Sequential()
        
        model.add(Dense(units=48,input_dim=self.state_size,activation="tanh"))
        
        model.add(Dense(units=self.action_size,activation="linear"))

        model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def remember(self,state,action,reward,next_state,done):
        
        # storage
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        
        # acting: explore or exploit
        if self.epsilon >= random.uniform(0,1):
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        
    
    def replay(self,batch_size):
        
        # training
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in mini_batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target,verbose=0)
                
    
    def adaptiveEGreedy(self):
        
        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

if __name__ == "__main__":
    
    # initialize env and agent
    env = gym.make("CartPole-v0")
    agent = DQLAgent(env)
    
    batch_size = 16
    episodes = 50
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        
        state = np.reshape(state,[1,4])
        
        timet = 0
        
        while True:
            
            # act
            action = agent.act(state) # select an action
            
            
            # step
            next_state,reward,done,_ = env.step(action)
            next_state = np.reshape(next_state,[1,4])
            
            # remember / storage
            agent.remember(state,action,reward,next_state,done)
            
            # update state
            state = next_state
            
            # replay 
            agent.replay(batch_size)
            
            # adjust epsilon 
            agent.adaptiveEGreedy()
            
            timet += 1
            
            if done:
                print("Episode: {}, Time: {}".format(e,timet))
                break




#%%

trained_model = agent

state = env.reset()
state = np.reshape(state,[1,4])
timet = 0
while True:
    env.render()
    action = trained_model.act(state)
    next_state,reward,done,_ = env.step(action)
    next_state = np.reshape(next_state,[1,4])
    state = next_state
    timet += 1
    print(timet)
    time.sleep(0.2)
    if done:
        break
print("Done")
        

#%%













































#%%