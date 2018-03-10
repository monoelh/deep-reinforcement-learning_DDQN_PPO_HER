#######################################################################################
# PPO A2C-style- Learning framework to play around with 
# Author: Manuel Hass
# 2018
# 
# *uses mlp_framework.py as model framework 
#  
#######################################################################################
#!!!!!!!!!!! quite messy 'quick & dirty'  code !!!!!!!!!!!!!!!!!

### imports
import numpy as np
import gym
import time 
import mlp_framework as nn

class ppo:
	'''
	ppo loss 
	'''
	def __init__(self):
		self.REWARD = 1
		self.PRED_REW = 1
		self.OLD_PROB = 1

		self.ADVANTAGE = self.REWARD - self.PRED_REW
		self.eps = 1e-10

		self.clipper = 0.2

	def ppo_loss(self,yt,y,deriv=True):
		probability = np.sum(yt*y)
		old_probability = np.sum(yt*self.OLD_PROB)
		
		ratio = probability / (old_probability+self.eps)
		clipped = np.clip(ratio,1-self.clipper,1+self.clipper)*self.ADVANTAGE
		term2 = np.min((ratio*self.ADVANTAGE,clipped),axis=1)
		term2 = np.mean(term2)
		loss = -np.log(probability+self.eps) * term2
		#print('loss: ',loss.shape)
		return  loss

'''
loss_obj = ppo()
loss = loss_obj.ppo_loss
print(loss(0,0))
loss_obj.REWARD = 10
print(loss(0,0))
''' and None

start = time.perf_counter()


######################## model ####################################
### create env
env = gym.make('CartPole-v0').env

### create layers
INPUT_SHAPE = env.observation_space.shape[0]
OUTPUT_SHAPE = env.action_space.n

A1 = nn.layer(INPUT_SHAPE,64,no_bias=True)
A2 = nn.layer(64,64,no_bias=True)
AOUT = nn.layer(64,OUTPUT_SHAPE,no_bias=True) # action out
AOUT.f = nn.f_softmax

L1 = nn.layer(INPUT_SHAPE,64)
L2 = nn.layer(64,64)
LOUT = nn.layer(64,1)# value out
LOUT.f = nn.f_iden

### create models
policy_model = nn.mlp([A1,A2,AOUT]) # policy model 
policyloss = ppo()
policy_model.erf = policyloss.ppo_loss 

value_model = nn.mlp([L1,L2,LOUT]) # value model

#use elu activation 
for L in value_model.Layerlist:
	L.f = nn.f_elu
value_model.Layerlist[-1].f = nn.f_iden

MAX_EPISODES = 2000
POLICY_STEPS = 5
VALUE_STEPS = 5

GAMMA = 0.98
LAMBDA = 0.96


########################## training ###########################
reward_log = []
for I in range(MAX_EPISODES):
	done = False
	current_state = env.reset()
	episode_reward = []
	episode_batch = [[],[],[]]
	while not done:
		prediction = policy_model.infer(current_state[True,:])	
		probas = prediction[0]
		a = np.random.choice(env.action_space.n, p=probas)
		actions = np.zeros(prediction.shape[1])
		actions[a] = 1

		next_state, reward, done, _ = env.step(a)

		episode_reward += [reward]
		episode_batch[0] += [current_state]
		episode_batch[1] += [actions]
		episode_batch[2] += [prediction]
		current_state = next_state

		if done:
			values =  value_model.infer(np.array(episode_batch[0]))
			values = np.insert(values,0,np.zeros((1,1)))
			reward_log += [np.sum(episode_reward)]
			if (I+1) % 10 ==  0 :
				print((I+1),'th episode finished after {} timesteps with reward {} --- 10eps avg: {}'.format(len(episode_reward),np.sum(episode_reward),np.mean(reward_log[-50:])))
			## this computes targets, values and advantages. check back with loss and GAE
			transformed_reward = []
			advantage = 0.
			advantage_list = []
			for i in reversed(range(len(episode_reward))):				
				td =  episode_reward[i] * GAMMA * values[i+1] - values[i]
				advantage = td + GAMMA * LAMBDA * advantage
				advantage_list += [advantage]
				transformed_reward += [(advantage + values[i])]
			
			transformed_reward = np.array(list(reversed(transformed_reward)))
			
			advantage_list = np.array(list(reversed(advantage_list)))
			advantage_list -= np.mean(advantage_list)
			advantage_list /= (np.std(advantage_list)+1e-10)
			'''
			# not GAE
			for i in range(len(episode_reward)):
				RT = episode_reward[i]

				for j in range(i+1,len(episode_reward)):
					RT += episode_reward[j] * (LAMBDA * GAMMA)**j 
				episode_reward[i] = RT #+  value_model.infer((episode_batch[0][j])[True,:]) *(LAMBDA * GAMMA)**(len(episode_reward)-1)
			'''
			episode_reward = transformed_reward
			break
		#### 


	STATE, ACTION, PREDICTION, REWARD = np.array(episode_batch[0]),np.array(episode_batch[1]),np.array(episode_batch[2]),np.array(episode_reward)

	REWARD = REWARD.reshape(-1)

	old_prob = PREDICTION
	predicted_reward = value_model.infer(STATE)
	policyloss.REWARD = REWARD
	policyloss.PRED_REW = predicted_reward
	policyloss.OLD_PROB = old_prob
	policyloss.ADVANTAGE = advantage_list
	policy_model.erf = policyloss.ppo_loss
	
	counter = int(reward_log[-1] * 2)
	
	for step in range(POLICY_STEPS):
		policy_model.train(STATE,ACTION)
	#print('policy update')
	for step in range(VALUE_STEPS):
		value_model.train(STATE,REWARD[:,True])
	#print('value update')
print('time elapsed: ',time.perf_counter()-start,'s')
#########################################################################

#################### plotting ###########################################
import matplotlib.pyplot as plt
import pandas as pd
rewdata = pd.Series(reward_log)
plt.figure(figsize=(14,8))
rewdata.plot(alpha=0.1,color='b')
rewdata.rolling(window=100).mean().plot(style='g',alpha=.9)
#rewdata.rolling(window=50).mean().plot(style='b',alpha=.7)
rewdata.rolling(window=20).mean().plot(style='r',alpha=.5)
plt.title('reward over episodes')
plt.legend()
plt.grid()
plt.show()

'''
plt.figure()
plt.plot(range(len(reward_log)),reward_log)
plt.show()

'''
