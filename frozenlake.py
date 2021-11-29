# -*- coding: utf-8 -*-
import gym
import numpy as np

env = gym.make('FrozenLake-v1')
env.monitor.start('tmp/Frozenlake-0.2', force= True)
# initialize Q-Table
Q = np.zeros([env.observation_space.n,env.action_space.n])

# set learning parameter
lr = .85
y = 0.99
num_episodes = 3000

# create lists to contain total rewards and steps per episode

rList = []
sList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    sList=[]
    # The Q-Table learning algorithm
    while not d and j<250:
        j+=1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(5./(i+1)))

        # Get new state and reward from environment
        s1,r,d,_ = env.step(a)

        # Get negative reward every step
        if r==0 :
            r=-0.001

        # Q-Learning
        Q[s,a]= Q[s,a]+lr*(r+y* np.max(Q[s1,:])-Q[s,a])
        s=s1
        rAll=rAll+r
        sList.append(s)

    rList.append(rAll)
    if r==1 :
        print(sList)
    print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, j,rAll ,np.mean(rList)))


env.monitor.close()

print ("Final Q-Table Values")
print ("          left          down          right          up")
print (Q)